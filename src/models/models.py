import torch
import torch.nn as nn
from .building_blocks import EGNNLayer, EvoformerBlock, FeatureProjection, EdgeUpdateLayer


class ProteinBindingGNN(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 1305,
        edge_input_dim: int = 4,
        hidden_dim: int = 256,
        num_egnn_layers: int = 3,
        num_evoformer_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        update_coords: bool = True,
        num_recycles: int = 3,  # R — how many Evoformer refinement passes
        alpha: float = 0.3,          
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_recycles = num_recycles
        self.alpha = nn.Parameter(torch.tensor(alpha))  # learnable recycling weight

        # --- Projections ---
        self.node_proj = FeatureProjection(node_input_dim, hidden_dim, dropout)
        self.edge_proj = FeatureProjection(edge_input_dim, hidden_dim, dropout)

        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                update_coords=update_coords,
                dropout=dropout,
            )
            for _ in range(num_egnn_layers)
        ])
        self.egnn_edge_updates = nn.ModuleList([
            EdgeUpdateLayer(hidden_dim, dropout)
            for _ in range(num_egnn_layers)
        ])

        # --- Evoformer (runs R times — attention refinement) ---
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=hidden_dim * 4,
                dropout=dropout,
                use_edge_bias=True,
            )
            for _ in range(num_evoformer_blocks)
        ])
        self.evoformer_edge_updates = nn.ModuleList([
            EdgeUpdateLayer(hidden_dim, dropout)
            for _ in range(num_evoformer_blocks)
        ])

        # --- Recycling signal projection ---
        self.recycle_proj = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        )

        # --- Output head ---
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    # Stage 1 — geometry encoding (called once per forward)
    # ------------------------------------------------------------------

    def _encode_geometry(self, h, pos, edge_index, edge_h):
        """
        Runs EGNN layers once. Returns updated h, pos, edge_h.
        These are fixed for all recycling passes — geometry doesn't change.
        """
        for egnn, edge_upd in zip(self.egnn_layers, self.egnn_edge_updates):
            h, pos = egnn(h, pos, edge_index, edge_h)
            edge_h = edge_upd(edge_h, h, edge_index)
        return h, pos, edge_h

    # ------------------------------------------------------------------
    # Stage 2 — attention refinement (called R times per forward)
    # ------------------------------------------------------------------

    def _refine_attention(
        self,
        h: torch.Tensor,           # (N, H) geometry-encoded features
        edge_index: torch.Tensor,
        edge_h: torch.Tensor,
        prev_h: torch.Tensor,      # (N, H) full h from previous pass — not logits
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refines attention using recycled signals.
        prev_h carries the full attention context from the previous pass.
        recycle_proj learns how much of the previous state to mix in —
        it acts as a learned gate, not just a linear rescaling.
        """
        recycle_signal = self.recycle_proj(prev_h)  # (N, H) → (N, H)
        alpha = torch.sigmoid(self.alpha)  # ensure alpha is between 0 and 1
        h = h + alpha * recycle_signal                       # residual injection

        for evo, edge_upd in zip(self.evoformer_blocks, self.evoformer_edge_updates):
            h = evo(h, edge_index, edge_h)
            edge_h = edge_upd(edge_h, h, edge_index)

        logits = self.output_head(self.final_norm(h)).squeeze(-1)  # (N,)
        return h, edge_h, logits   # return h — this is what gets recycled

    # ------------------------------------------------------------------
    # Forward — wires both stages with recycling loop
    # ------------------------------------------------------------------

    def forward(self, data) -> torch.Tensor:
        h          = data.x
        pos        = data.pos
        edge_index = data.edge_index
        edge_h     = self.edge_proj(data.edge_attr)
        h          = self.node_proj(h)

        # Stage 1 — geometry, runs once
        h_geom, pos, edge_h_geom = self._encode_geometry(
            h, pos, edge_index, edge_h
        )

        # Stage 2 — recycling loop
        prev_h  = torch.zeros_like(h_geom)
        edge_in = edge_h_geom

        for recycle_idx in range(self.num_recycles):
            last_pass = (recycle_idx == self.num_recycles - 1)

            # Pass 0: start from geometry encoding
            # Pass 1+: continue evolving from previous refined state
            if recycle_idx == 0:
                h_in = h_geom if last_pass else h_geom.detach()
            else:
                h_in = prev_h if last_pass else prev_h.detach()

            prev_in  = prev_h  if last_pass else prev_h.detach()
            edge_in_ = edge_in if last_pass else edge_in.detach()

            h_refined, edge_refined, logits = self._refine_attention(
                h_in, edge_index, edge_in_, prev_in
            )

            prev_h  = h_refined
            edge_in = edge_refined

        return logits

    def predict(self, data) -> torch.Tensor:
        """Inference — returns probabilities, no gradient needed."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(data))



