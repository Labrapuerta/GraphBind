# GraphBind-UWU

Ligand binding site prediction in proteins using graph neural networks.

GraphBind-UWU is a GNN architecture built on top of PyTorch Geometric. It provides the source code, documentation, trained models and a robust explanation on the theory behind it.

The model is designed to predict ligand binding sites in proteins, which is a crucial task in drug discovery and structural biology. It takes as input a graph representation of a protein structure, where nodes represent amino acid residues and edges represent spatial proximity or chemical interactions. The model then processes this graph through a series of EGNN layers for geometry encoding, followed by Evoformer-style attention blocks with recycling to refine the predictions iteratively.

## Quick Start

### Installation

**For CPU or Apple Silicon (MPS):**
```bash
conda env create -f environment.yml
conda activate graphbind
```

**For NVIDIA GPUs (CUDA):**
```bash
conda env create -f environment_cuda.yml
conda activate graphbind-cuda
```

### Running Predictions

```bash
python predict.py --pdb path/to/your/protein.pdb
```

This will:
1. Load the trained model
2. Download the checkpoint from Hugging Face on first run (`ManuelLabra/GraphBindUwU` → `graph_bind_uwu.pt`)
3. Preprocess the PDB file (ESM-2 embeddings auto-download on first run ~2.5GB)
4. Predict binding sites
5. Open PyMOL with visualization (white = non-binding, red = binding)

All outputs are saved to a temp folder (printed at runtime).

**Options:**
```bash
python predict.py --pdb protein.pdb --threshold 0.6    # Custom threshold (default: 0.5)
python predict.py --pdb protein.pdb --no-pymol         # Skip PyMOL visualization
python predict.py --pdb protein.pdb --device mps       # Use Apple Silicon GPU
python predict.py --pdb protein.pdb --device cuda      # Use NVIDIA GPU
python predict.py --pdb protein.pdb --checkpoint ./local.pt  # Use a local checkpoint instead of Hugging Face
```

**Outputs:**
- `<protein>_binding.pdb` - PDB with binding probability in B-factor column
- `<protein>_predictions.csv` - Full predictions table
- `visualize_binding.pml` - PyMOL visualization script

### Requirements

- Python 3.10
- PyTorch 2.5
- PyMOL (for visualization)
- DSSP (`mkdssp` binary for hydrogen bond detection)
- Internet access on first run to download the checkpoint from Hugging Face

All dependencies are handled by the conda environment files.

---

## Dataset
We use the [BioLiP](https://zhanggroup.org/BioLiP/) database, which contains high-quality annotations of ligand binding sites in protein structures. The dataset is preprocessed to extract relevant representations of each family and split into training, validation and test sets. 

## Preprocessing
To prepare the data for training, we convert protein structures into graph representations. Each node in the graph corresponds to an amino acid residue, and edges are defined based on spatial proximity (e.g., residues within a certain distance threshold). Node features include ESM2 embeddings, one-hot encodings of amino acid types, and biochemical properties. Edge features include distance, bond type, contact probability, and Coulomb interactions.

To generate the graph from the raw protein structure, we use the `preprocess_protein` function defined in `src/data/preprocessing.py`. This function takes a PDB file as input and outputs a graph object compatible with PyTorch Geometric. In case you want to preprocess your own protein structures, you can use the following colab notebook: [Preprocessing Notebook](https://colab.research.google.com/drive/1irGNqG7A5CT08rTQ9oldHe4iksVgWJU_?usp=sharing) (link to be updated).



## Architecture :)

Equivariant GNN with Evoformer-style Attention & Recycling
for Protein Binding Site Prediction
===================================================================================
ARCHITECTURE OVERVIEW
===================================================================================

```python
                              INPUT
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
  ┌───────────┐           ┌───────────┐           ┌───────────┐
  │  Node     │           │   Edge    │           │  Position │
  │ Features  │           │ Features  │           │   (pos)   │
  │  (1305,)  │           │   (4,)    │           │   (3,)    │
  │ ESM2+AA+  │           │ dist,type │           │    CA     │
  │  biochem  │           │ cont,coul │           │  coords   │
  └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
        │                       │                       │
        ▼                       ▼                       │
  ┌───────────┐           ┌───────────┐                 │
  │ node_proj │           │ edge_proj │                 │
  │ 1305 → H  │           │  4 → H    │                 │
  └─────┬─────┘           └─────┬─────┘                 │
        │                       │                       │
        └───────────┬───────────┘                       │
                    │                                   │
                    ▼                                   ▼
┌═══════════════════════════════════════════════════════════════════┐
║                                                                   ║
║   STAGE 1: GEOMETRY ENCODING (runs ONCE)                          ║
║                                                                   ║
║   ┌─────────────────────────────────────────────────────────┐     ║
║   │                   EGNN BLOCK × 3                        │     ║
║   │  ┌─────────────────────────────────────────────────┐    │     ║
║   │  │              Message Passing                    │    │     ║
║   │  │   ┌────────────────────────────────────────┐    │    │     ║
║   │  │   │  m_ij = MLP([h_i ∥ h_j ∥ x_i-x_j        |    │    │     ║
║   │  │   │               e_ij])                   │    │    │     ║
║   │  │   └────────────────────────────────────────┘    │    │     ║
║   │  │                      │                          │    │     ║
║   │  │   ┌──────────────────┴─────────────────────┐    │    │     ║
║   │  │   │         Invariant     Equivariant      │    │    │     ║
║   │  │   │            ▼              ▼            │    │    │     ║
║   │  │   │      Σ_j m_ij      Σ_j(x_i-x_j)·φ(m)   │    │    │     ║
║   │  │   └──────────────────┬─────────────────────┘    │    │     ║
║   │  │                      ▼                          │    │     ║
║   │  │   ┌────────────────────────────────────────┐    │    │     ║
║   │  │   │  h_i' = h_i + Σ_j m_ij   (residual)    │    │    │     ║
║   │  │   │  x_i' = x_i + Σ_j(x_i-x_j)·φ(m)/N      │    │    │     ║
║   │  │   └────────────────────────────────────────┘    │    │     ║
║   │  └─────────────────────────────────────────────────┘    │     ║
║   │                          │                              │     ║
║   │                          ▼                              │     ║
║   │  ┌─────────────────────────────────────────────────┐    │     ║
║   │  │         EdgeUpdateLayer                         │    │     ║
║   │  │   e'_ij = MLP([e_ij ∥ h_i' ∥ h_j'])              │    │     ║
║   │  └─────────────────────────────────────────────────┘    │     ║
║   └─────────────────────────────────────────────────────────┘     ║
║                          │                                        ║
║                          ▼                                        ║
║                    h_geom, pos, edge_h_geom                       ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
                           │
                           ▼
┌═══════════════════════════════════════════════════════════════════┐
║                                                                   ║
║   STAGE 2: ATTENTION REFINEMENT (runs R times - RECYCLING)        ║
║                                                                   ║
║   ┌─────────────────────────────────────────────────────────────┐ ║
║   │                                                             │ ║
║   │    prev_h ──────────────────────────┐                       │ ║
║   │    (from previous recycle pass)     │                       │ ║
║   │                                     ▼                       │ ║
║   │                            ┌────────────────┐               │ ║
║   │                            │  recycle_proj  │               │ ║
║   │                            │   H → H        │               │ ║
║   │                            └───────┬────────┘               │ ║
║   │                                    │                        │ ║
║   │    h_geom ─────────────────────────┤                        │ ║
║   │                                    ▼                        │ ║
║   │                        ┌───────────────────────┐            │ ║
║   │                        │ h = h + α·recycle(h') │            │ ║
║   │                        │   (α is learnable)    │            │ ║
║   │                        └───────────┬───────────┘            │ ║
║   │                                    │                        │ ║
║   │                                    ▼                        │ ║
║   │   ┌─────────────────────────────────────────────────────┐   │ ║
║   │   │         EVOFORMER BLOCK × 4                         │   │ ║
║   │   │                                                     │   │ ║
║   │   │   h ─────┬────────────────────────────┐             │   │ ║
║   │   │          │                            │             │   │ ║
║   │   │          ▼                            │             │   │ ║
║   │   │   ┌─────────────┐                     │             │   │ ║
║   │   │   │  LayerNorm  │                     │             │   │ ║
║   │   │   └──────┬──────┘                     │             │   │ ║
║   │   │          ▼                            │             │   │ ║
║   │   │   ┌─────────────┐                     │ (skip)      │   │ ║
║   │   │   │ Multi-Head  │   edge_bias         │             │   │ ║
║   │   │   │ Attention   │◄──────────          │             │   │ ║
║   │   │   │ (Q,K,V)     │                     │             │   │ ║
║   │   │   └──────┬──────┘                     │             │   │ ║
║   │   │          │◄───────────────────────────┘             │   │ ║
║   │   │          ▼   (+)                                    │   │ ║
║   │   │   h' = h + Attn(h)                                  │   │ ║
║   │   │                                                     │   │ ║
║   │   │   h' ────┬────────────────────────────┐             │   │ ║
║   │   │          ▼                            │             │   │ ║
║   │   │   ┌─────────────┐                     │             │   │ ║
║   │   │   │  LayerNorm  │                     │             │   │ ║
║   │   │   └──────┬──────┘                     │             │   │ ║
║   │   │          ▼                            │ (skip)      │   │ ║
║   │   │   ┌─────────────┐                     │             │   │ ║
║   │   │   │     FFN     │                     │             │   │ ║
║   │   │   │  H → 4H → H │                     │             │   │ ║
║   │   │   └──────┬──────┘                     │             │   │ ║
║   │   │          │◄───────────────────────────┘             │   │ ║
║   │   │          ▼   (+)                                    │   │ ║
║   │   │   h'' = h' + FFN(h')                                │   │ ║
║   │   └─────────────────────────────────────────────────────┘   │ ║
║   │                          │                                  │ ║
║   │                          ▼                                  │ ║
║   │   ┌─────────────────────────────────────────────────────┐   │ ║
║   │   │         EdgeUpdateLayer                             │   │ ║
║   │   │   e'_ij = MLP([e_ij ∥ h_i'' ∥ h_j''])                │   │ ║
║   │   └─────────────────────────────────────────────────────┘   │ ║
║   │                          │                                  │ ║
║   └──────────────────────────┼──────────────────────────────────┘ ║
║                              │                                    ║
║                              ▼                                    ║
║                     ┌────────────────┐                            ║
║                     │   Final Norm   │                            ║
║                     └───────┬────────┘                            ║
║                             │                                     ║
║                             ▼                                     ║
║                     ┌────────────────┐                            ║
║                     │  Output Head   │                            ║
║                     │  H → H/2 → 1   │                            ║
║                     └───────┬────────┘                            ║
║                             │                                     ║
║                             ▼                                     ║
║                          logits ──────────────┐                   ║
║                             │                 │                   ║
║           ┌─────────────────┘                 │                   ║
║           │                                   │                   ║
║           ▼                                   ▼                   ║
║   ┌───────────────┐                  ┌────────────────┐           ║
║   │ if last pass: │                  │  h_refined →   │           ║
║   │    return     │                  │  prev_h for    │           ║
║   └───────────────┘                  │  next recycle  │           ║
║                                      └────────┬───────┘           ║
║                                               │                   ║
║                              ◄────────────────┘                   ║
║                        (loop R times)                             ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
                           │
                           ▼
                   ┌───────────────┐
                   │    Sigmoid    │
                   │  (inference)  │
                   └───────────────┘

```
===================================================================================
KEY FEATURES
===================================================================================

1. TWO-STAGE ARCHITECTURE
   - Stage 1 (EGNN): Geometry encoding with equivariant message passing
   - Stage 2 (Evoformer): Attention-based refinement with recycling

2. RECYCLING MECHANISM (inspired by AlphaFold2)
   - Runs attention refinement R times (default: 3)
   - Each pass refines predictions using previous hidden states
   - Learnable α parameter controls recycling strength
   - Gradients only flow through last pass (memory efficient)

3. EDGE UPDATES
   - Both EGNN and Evoformer update edge features
   - Edge bias informs attention computation

4. INPUT FEATURES
   - Node: ESM2 embeddings (1280) + one-hot AA (20) + biochem (5) = 1305
   - Edge: distance, bond type, contact prob, coulomb = 4
   - Position: CA atom coordinates (3)

===================================================================================
HYPERPARAMETERS (defaults)
===================================================================================

  hidden_dim:           256      # H - hidden dimension
  num_egnn_layers:      3        # EGNN blocks in Stage 1
  num_evoformer_blocks: 4        # Evoformer blocks per recycle
  num_heads:            8        # Attention heads
  num_recycles:         3        # R - recycling passes
  alpha:                0.3      # Initial recycling weight (learnable)
  dropout:              0.1

```

