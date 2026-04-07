import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple


# =============================================================================
# BUILDING BLOCK 1: Feature Projection (like Dense layers in TF)
# =============================================================================

class FeatureProjection(nn.Module):
    """
    Projects raw features to hidden dimension.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        super().__init__() 

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        x = self.dropout(x)
        return x
    
# =============================================================================
# Edge update projection 
# =============================================================================

class EdgeUpdateLayer(nn.Module):
    """
    Updates edge features using the current node features of both endpoints.
    Runs in parallel with the node update layers.
 
    h_src and h_dst are the *current* (post-update) node features,
    so edges always reflect the latest node context.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),  # e + h_src + h_dst
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(
        self,
        edge_attr: torch.Tensor,    # (E, H) current edge features
        h: torch.Tensor,            # (N, H) current node features
        edge_index: torch.Tensor,   # (2, E)
    ) -> torch.Tensor:
        src, dst = edge_index
        # Gather endpoint features for each edge
        edge_input = torch.cat([edge_attr, h[src], h[dst]], dim=-1)  # (E, 3H)
        edge_attr = edge_attr + self.dropout(self.mlp(edge_input))   # residual
        return self.norm(edge_attr)


# =============================================================================
# BUILDING BLOCK 2: E(3) Equivariant Graph Conv Layer
# =============================================================================

class EGNNLayer(MessagePassing):
    """
    E(n) Equivariant Graph Neural Network Layer.

    Key insight for equivariance:
    - Node features (h) are INVARIANT (don't change with rotation)
    - Positions (x) are EQUIVARIANT (rotate with the molecule)
    This layer updates both h and x while maintaining these properties.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, update_coords: bool = True, activation: str = "SiLU", dropout: float = 0.1,):
        # aggr='add' means: aggregate messages by summing (like tf.math.segment_sum)

        super().__init__(aggr='add') # Pending: can experiment with 'mean' or 'max' for different effects

        self.hidden_dim = hidden_dim
        self.update_coords = update_coords # Boool
        self.activation = activation ## Pending, currently only SiLu

        # Message MLP: takes [h_i, h_j, distance, edge_features]
        # Input: hidden_dim*2 + 1 + edge_dim


        ### We can modify the activation function here ..., SiLU (Swish) is a smooth activation that can improve performance in some cases
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),  # Swish activation, popular in recent architectures
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),    
        )

        # Coordinate update MLP: outputs a scalar weight for direction vector
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),  # Output scalar
            )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,           # (N, hidden_dim) node features
        pos: torch.Tensor,         # (N, 3) coordinates
        edge_index: torch.Tensor,  # (2, E) edges
        edge_attr: torch.Tensor,   # (E, edge_dim) edge features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (h', pos')
        """
        # Compute pairwise distances (INVARIANT - doesn't change with rotation)
        row, col = edge_index  # source, target indices

        # Relative position vector (EQUIVARIANT)
        rel_pos = pos[row] - pos[col]  # (E, 3)

        # Distance (INVARIANT) - safe for gradients
        dist = torch.norm(rel_pos, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, 1)

        # propagate() calls message() then aggregate() - PyG handles this
        # It's like tf.math.unsorted_segment_sum but more convenient
        h_out, coord_update = self.propagate(
            edge_index,
            h=h,
            dist=dist,
            rel_pos=rel_pos,
            edge_attr=edge_attr,
        )

        h_out = h + self.dropout(h_out) # Residual connection with dropout for regularization
        h_out = self.norm(h_out) 

        # Update coordinates (equivariant update)
        if self.update_coords and coord_update is not None:
            # Normalize by number of neighbors to stabilize
            # This maintains equivariance: rotating input rotates output
            pos = pos + coord_update

        return h_out, pos

    def message(
        self,
        h_i: torch.Tensor,      # (E, H) features of source nodes
        h_j: torch.Tensor,      # (E, H) features of target nodes
        dist: torch.Tensor,     # (E, 1) distances
        rel_pos: torch.Tensor,  # (E, 3) relative positions
        edge_attr: torch.Tensor # (E, edge_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute message from j to i.

        In TensorFlow you'd write:
            inputs = tf.concat([h_i, h_j, dist, edge_attr], axis=-1)
            messages = self.message_mlp(inputs)
        """
        # Concatenate all invariant features
        inputs = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)

        # Message (invariant)
        m_ij = self.message_mlp(inputs)  # (E, H)

        # Coordinate message (equivariant)
        if self.update_coords:
            # Scalar weight from message, multiply by direction
            coord_weight = self.coord_mlp(m_ij)  # (E, 1)
            coord_msg = rel_pos * coord_weight   # (E, 3) - equivariant!
        else:
            coord_msg = torch.zeros_like(rel_pos)

        # Return both for aggregation
        # We store coord_msg in a way that propagate can handle
        self._coord_msg = coord_msg
        return m_ij

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: int = None):
        """
        Aggregate messages to nodes.

        TensorFlow equivalent:
            tf.math.unsorted_segment_sum(inputs, index, num_segments)
        """
        # Aggregate node messages
        h_agg = super().aggregate(inputs, index, dim_size=dim_size)

        # Aggregate coordinate updates
        if hasattr(self, '_coord_msg'):
            coord_agg = super().aggregate(self._coord_msg, index, dim_size=dim_size)
        else:
            coord_agg = None

        return h_agg, coord_agg

    def update(self, agg_out, h):
        """
        Update node features with aggregated messages.

        agg_out: (h_agg, coord_agg) from aggregate()
        h: original node features
        """
        h_agg, coord_agg = agg_out

        # Combine original features with aggregated messages
        h_new = self.node_mlp(torch.cat([h, h_agg], dim=-1))

        return h_new, coord_agg


# =============================================================================
# BUILDING BLOCK 3: Evoformer-style Attention Block
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with edge bias.

    Like the attention in Evoformer/AlphaFold:
    - Q, K, V projections
    - Optional edge features as attention bias
    - Multiple heads for capturing different relationships

    TensorFlow equivalent: tf.keras.layers.MultiHeadAttention
    but we add edge bias capability.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_bias: bool = True
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k) for scaled dot-product

        # Q, K, V projections (like tf.keras.layers.Dense)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Edge feature projection for attention bias
        self.use_edge_bias = use_edge_bias
        if use_edge_bias:
            self.edge_proj = nn.Linear(hidden_dim, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,                    # (N, H) node features
        edge_index: torch.Tensor,           # (2, E) edge indices
        edge_attr: Optional[torch.Tensor] = None,  # (E, H) edge features
    ) -> torch.Tensor:
        """
        Self-attention over graph nodes with optional edge bias.

        For proteins: nodes=residues, edges=contacts/bonds
        Attention helps each residue "see" relevant context from other residues.
        """
        N = h.size(0)

        # Project to Q, K, V
        # Shape: (N, H) -> (N, num_heads, head_dim)
        Q = self.q_proj(h).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(h).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(h).view(N, self.num_heads, self.head_dim)

        # Compute attention scores: Q @ K^T / sqrt(d)
        # Shape: (N, num_heads, head_dim) @ (N, head_dim, num_heads)
        # We use einsum for clarity (like tf.einsum)
        # 'nhd,mhd->nmh' means: for each head, compute dot product between all pairs
        attn_scores = torch.einsum('nhd,mhd->nmh', Q, K) * self.scale  # (N, N, num_heads)

        # Add edge bias if using (Evoformer-style)
        if self.use_edge_bias and edge_attr is not None:
            # Project edge features to per-head bias
            edge_bias = self.edge_proj(edge_attr)  # (E, num_heads)

            # Create sparse attention bias matrix
            row, col = edge_index
            # Only add bias where edges exist
            attn_scores[row, col] = attn_scores[row, col] + edge_bias

        # Softmax over source nodes (dim=1, the 'm' dimension)
        attn_weights = F.softmax(attn_scores, dim=1)  # (N, N, num_heads)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # 'nmh,mhd->nhd' means: weighted sum of V vectors
        out = torch.einsum('nmh,mhd->nhd', attn_weights, V)  # (N, num_heads, head_dim)

        # Reshape and project output
        out = out.reshape(N, self.hidden_dim)  # (N, H)
        out = self.out_proj(out)

        return out


class EvoformerBlock(nn.Module):
    """
    Single Evoformer-style block with:
    1. Multi-head self-attention + residual
    2. Feed-forward network + residual

    This is the "Transformer block" pattern used in:
    - Original Transformer (Vaswani et al.)
    - BERT, GPT
    - AlphaFold's Evoformer (with pair updates)

    The key insight: residual connections let gradients flow and
    prevent loss of information in deep networks.

    TensorFlow equivalent structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_dim: int = None,  # Usually 4x hidden_dim
        dropout: float = 0.1,
        use_edge_bias: bool = True
    ):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = hidden_dim * 4  # Standard transformer ratio

        # Pre-norm (more stable than post-norm for deep networks)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_edge_bias=use_edge_bias
        )

        # Feed-forward network (2-layer MLP with expansion)
        # This is like: Dense(4H) -> GELU -> Dense(H)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),  # GELU is smoother than ReLU, used in modern transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pre-LN Transformer block (more stable for deep networks).

        h: (N, H) node features
        Returns: (N, H) updated node features
        """
        # Self-attention with residual
        # Pattern: x = x + Sublayer(Norm(x))
        h = h + self.dropout(
            self.attention(
                self.norm1(h),
                edge_index,
                edge_attr
            )
        )

        # FFN with residual
        h = h + self.ffn(self.norm2(h))

        return h




