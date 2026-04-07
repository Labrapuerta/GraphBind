#!/usr/bin/env python3
"""
GraphBind-UWU Prediction Script

Predicts protein binding sites from a PDB file using the trained GraphBind model.
Visualizes results in PyMOL with binding residues colored.

Usage:
    python predict.py --pdb <path/to/protein.pdb> [--threshold 0.5] [--no-pymol]
"""

import argparse
import sys
import subprocess
import hashlib
import tempfile
from pathlib import Path

import torch
import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, DSSP, Selection
from Bio.Data.PDBData import protein_letters_3to1 as three_to_one
from torch_geometric.data import Data
from huggingface_hub import hf_hub_download

from src.models.models import ProteinBindingGNN
from model.config import CONFIG

# Constants for graph construction
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}
EDGE_TYPE_MAP = {"peptide": 0, "vdw": 1, "hbond": 2}
VDW_CUTOFF = 8.0
HBOND_ENERGY_THRESHOLD = -0.5
PEPTIDE_BOND_CUTOFF = 1.5
DIELECTRIC_CONSTANT = 4.0
HF_REPO_ID = "ManuelLabra/GraphBindUwU"
HF_CHECKPOINT_FILENAME = "graph_bind_uwu.pt"
HF_CACHE_DIR = Path.home() / ".cache" / "graphbind_uwu"

# Physicochemical constants
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}
FORMAL_CHARGE = {
    'ALA': 0, 'ARG': 1, 'ASN': 0, 'ASP': -1, 'CYS': 0,
    'GLN': 0, 'GLU': -1, 'GLY': 0, 'HIS': 0, 'ILE': 0,
    'LEU': 0, 'LYS': 1, 'MET': 0, 'PHE': 0, 'PRO': 0,
    'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0, 'VAL': 0
}
ISOELECTRIC = {
    'ALA': 6.00, 'ARG': 10.76, 'ASN': 5.41, 'ASP': 2.77, 'CYS': 5.07,
    'GLN': 5.65, 'GLU': 3.22, 'GLY': 5.97, 'HIS': 7.59, 'ILE': 6.02,
    'LEU': 5.98, 'LYS': 9.74, 'MET': 5.74, 'PHE': 5.48, 'PRO': 6.30,
    'SER': 5.68, 'THR': 5.60, 'TRP': 5.89, 'TYR': 5.66, 'VAL': 5.96
}


class ESMEmbedder:
    """ESM-2 embedder using torch.hub for automatic model download."""
    
    def __init__(self, device: str = "cpu", cache_dir: str = ".esm_cache"):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._model = None
        self._alphabet = None
        self._batch_converter = None
    
    def _load_model(self):
        if self._model is None:
            print("[ESM2] Loading model via torch.hub (this may download ~2.5GB on first run)...")
            self._model, self._alphabet = torch.hub.load(
                "facebookresearch/esm:main", 
                "esm2_t33_650M_UR50D"
            )
            self._batch_converter = self._alphabet.get_batch_converter()
            self._model = self._model.eval().to(self.device)
            print("[ESM2] Model loaded.")
    
    def _cache_path(self, sequence: str) -> Path:
        key = hashlib.md5(sequence.encode()).hexdigest()
        return self.cache_dir / f"{key}.pt"
    
    def get_embeddings(self, sequence: str) -> tuple:
        """Get ESM-2 embeddings and contact map for a sequence."""
        # Check cache first
        cache_file = self._cache_path(sequence)
        if cache_file.exists():
            cached = torch.load(cache_file, map_location="cpu")
            return cached["embeddings"], cached["contacts"]
        
        self._load_model()
        
        _, _, batch_tokens = self._batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(self.device)
        
        with torch.no_grad():
            results = self._model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=True,
            )
        
        L = len(sequence)
        embeddings = results["representations"][33][0, 1:L+1].cpu()  # (L, 1280)
        contacts = results["contacts"][0, :L, :L].cpu()              # (L, L)
        
        # Cache results
        torch.save({"embeddings": embeddings, "contacts": contacts}, cache_file)
        
        return embeddings, contacts


class SimpleGraphBuilder:
    """Simplified protein graph builder for prediction."""
    
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", pdb_path)
        self.model = next(self.structure.get_models())
        
        self._raw_residues = [
            r for r in Selection.unfold_entities(self.model, "R")
            if "CA" in r
        ]
        
        self.sequence, self.valid_indices, self.skipped = self._extract_sequence()
        self.residues = [self._raw_residues[i] for i in self.valid_indices]
        self.res_to_idx = {r: i for i, r in enumerate(self.residues)}
        self._cb_coords = self._get_cb_coords()
        self._dssp_map = self._build_dssp_map()
        
        print(f"[GraphBuilder] {len(self.residues)} residues | sequence length: {len(self.sequence)}")
    
    def _extract_sequence(self):
        sequence, valid_indices, skipped = [], [], []
        for i, res in enumerate(self._raw_residues):
            resname = res.get_resname().strip()
            het, resseq, icode = res.get_id()
            if het.strip():
                skipped.append((i, res, "HETATM"))
                continue
            if resname not in STANDARD_AA:
                skipped.append((i, res, f"non-standard: {resname}"))
                continue
            try:
                one_letter = three_to_one[resname]
            except KeyError:
                skipped.append((i, res, "no one-letter code"))
                continue
            sequence.append(one_letter)
            valid_indices.append(i)
        return "".join(sequence), valid_indices, skipped
    
    def _get_cb_coords(self):
        coords = []
        for res in self.residues:
            atom = res["CB"] if "CB" in res else res["CA"]
            coords.append(atom.get_vector().get_array())
        return np.array(coords)
    
    def _build_dssp_map(self):
        dssp_map = {}
        for res in self.residues:
            chain_id = res.get_parent().id
            dssp_key = (chain_id, res.get_id())
            dssp_map[dssp_key] = self.res_to_idx[res]
        return dssp_map
    
    def _peptide_edges(self):
        edges = []
        for i in range(len(self.residues) - 1):
            r1, r2 = self.residues[i], self.residues[i + 1]
            if r1.get_parent().id != r2.get_parent().id:
                continue
            if "C" not in r1 or "N" not in r2:
                continue
            c_coord = r1["C"].get_vector().get_array()
            n_coord = r2["N"].get_vector().get_array()
            if np.linalg.norm(c_coord - n_coord) <= PEPTIDE_BOND_CUTOFF:
                edges.append((i, i + 1, 1.0, "peptide"))
        return edges
    
    def _vdw_edges(self):
        tree = cKDTree(self._cb_coords)
        edges = []
        for i, j in tree.query_pairs(r=VDW_CUTOFF):
            dist = np.linalg.norm(self._cb_coords[i] - self._cb_coords[j])
            weight = np.exp(-dist / VDW_CUTOFF)
            edges.append((i, j, weight, "vdw"))
        return edges
    
    def _hbond_edges(self):
        try:
            dssp = DSSP(self.model, self.pdb_path, dssp="mkdssp", file_type="PDB")
        except Exception as e:
            print(f"[Warning] DSSP failed: {e}. Skipping H-bond edges.")
            return []
        
        edges = []
        dssp_keys = list(dssp.keys())
        for idx, key in enumerate(dssp_keys):
            data = dssp[key]
            for offset_field, energy_field in [(6, 7), (8, 9)]:
                offset = data[offset_field]
                energy = data[energy_field]
                if energy >= HBOND_ENERGY_THRESHOLD or offset == 0:
                    continue
                partner_idx = idx + offset
                if not (0 <= partner_idx < len(dssp_keys)):
                    continue
                partner_key = dssp_keys[partner_idx]
                src_node = self._dssp_map.get(key)
                dst_node = self._dssp_map.get(partner_key)
                if src_node is not None and dst_node is not None:
                    edges.append((src_node, dst_node, abs(energy), "hbond"))
        return edges
    
    def _get_node_features(self, esm_embeddings: torch.Tensor) -> torch.Tensor:
        """Concatenate ESM embeddings with physicochemical features."""
        n = len(self.residues)
        
        # One-hot encoding (20 dim)
        aa_list = sorted(STANDARD_AA)
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
        one_hot = torch.zeros(n, 20)
        for i, res in enumerate(self.residues):
            resname = res.get_resname().strip()
            if resname in aa_to_idx:
                one_hot[i, aa_to_idx[resname]] = 1.0
        
        # Physicochemical features (5 dim)
        hydro = torch.tensor([[HYDROPHOBICITY.get(r.get_resname().strip(), 0.0)] for r in self.residues])
        charge = torch.tensor([[float(FORMAL_CHARGE.get(r.get_resname().strip(), 0))] for r in self.residues])
        iso = torch.tensor([[ISOELECTRIC.get(r.get_resname().strip(), 5.97)] for r in self.residues])
        
        # Side chain length
        sc_len = []
        for res in self.residues:
            if "CA" in res:
                ca = res["CA"].get_vector().get_array()
                sc_atoms = [a for a in res.get_atoms() if a.get_name() not in ("N", "CA", "C", "O")]
                if sc_atoms:
                    dists = [np.linalg.norm(a.get_vector().get_array() - ca) for a in sc_atoms]
                    sc_len.append([max(dists)])
                else:
                    sc_len.append([0.0])
            else:
                sc_len.append([0.0])
        sc_len = torch.tensor(sc_len, dtype=torch.float)
        
        # B-factor
        b_factors = []
        for res in self.residues:
            atoms = list(res.get_atoms())
            if atoms:
                b_factors.append([np.mean([a.get_bfactor() for a in atoms])])
            else:
                b_factors.append([0.0])
        b_factors = torch.tensor(b_factors, dtype=torch.float)
        
        return torch.cat([esm_embeddings, one_hot, hydro, charge, iso, sc_len, b_factors], dim=1)
    
    def build(self, esm_embeddings: torch.Tensor, contacts: torch.Tensor) -> Data:
        """Build the protein graph."""
        all_edges = self._peptide_edges() + self._vdw_edges() + self._hbond_edges()
        
        src = torch.tensor([e[0] for e in all_edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in all_edges], dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        
        weights = torch.tensor([e[2] for e in all_edges], dtype=torch.float)
        types = torch.tensor([EDGE_TYPE_MAP[e[3]] for e in all_edges], dtype=torch.float)
        
        zero = torch.zeros_like(weights)
        contacts_fw = contacts[src, dst] if contacts is not None else zero
        contacts_rv = contacts[dst, src] if contacts is not None else zero
        
        # Coulomb term
        charges = torch.tensor([float(FORMAL_CHARGE.get(r.get_resname().strip(), 0)) for r in self.residues])
        q_src, q_dst = charges[src], charges[dst]
        cb_coords = torch.tensor(self._cb_coords, dtype=torch.float)
        r = torch.norm(cb_coords[src] - cb_coords[dst], dim=1).clamp(min=1e-6)
        coulomb_fw = q_src * q_dst / (DIELECTRIC_CONSTANT * r ** 2)
        coulomb_rv = coulomb_fw  # symmetric
        
        scalar_fw = torch.stack([weights, types, contacts_fw, coulomb_fw], dim=1)
        scalar_rv = torch.stack([weights, types, contacts_rv, coulomb_rv], dim=1)
        edge_attr = torch.cat([scalar_fw, scalar_rv], dim=0)
        
        node_features = self._get_node_features(esm_embeddings)
        
        return Data(
            x=node_features,
            pos=cb_coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.residues)
        )


def load_model(checkpoint_path: str, device: torch.device) -> ProteinBindingGNN:
    """Load the trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    # Use config from checkpoint (trained model's actual architecture)
    # Fall back to CONFIG only if checkpoint doesn't have config
    if "config" in checkpoint:
        cfg = checkpoint["config"]
    else:
        cfg = CONFIG
        print("[Warning] Checkpoint has no config, using model/config.py defaults")
    
    model = ProteinBindingGNN(
        node_input_dim=cfg["node_input_dim"],
        edge_input_dim=cfg["edge_input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_egnn_layers=cfg["num_egnn_layers"],
        num_evoformer_blocks=cfg["num_evoformer_blocks"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        update_coords=cfg["update_coords"],
        num_recycles=cfg["num_recycles"],
        alpha=cfg["alpha"],
    ).to(device)
    
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def resolve_checkpoint_path(
    checkpoint_path: str | None,
    hf_repo_id: str,
    hf_filename: str,
    cache_dir: Path,
) -> Path:
    """Resolve a local checkpoint or download it from Hugging Face."""
    if checkpoint_path:
        local_path = Path(checkpoint_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local_path}")
        return local_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Config] Downloading checkpoint from Hugging Face: {hf_repo_id}/{hf_filename}")
    downloaded_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_filename,
        cache_dir=str(cache_dir),
    )
    return Path(downloaded_path).resolve()


def preprocess_pdb(pdb_path: str, device: str = "cpu") -> tuple:
    """Preprocess PDB file to create graph representation."""
    print(f"[Preprocessing] Building graph from {pdb_path}")
    builder = SimpleGraphBuilder(pdb_path)
    
    print("[Preprocessing] Running ESM2 for sequence embeddings...")
    esm = ESMEmbedder(device=device)
    embeddings, contacts = esm.get_embeddings(builder.sequence)
    
    print("[Preprocessing] Building final graph...")
    data = builder.build(embeddings, contacts)
    
    return data, builder


def get_residue_indices(pdb_path: str) -> list[int]:
    """Extract unique residue indices from PDB file in order of appearance."""
    residues = []
    seen = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                try:
                    res_num = int(line[22:26].strip())
                    if res_num not in seen:
                        residues.append(res_num)
                        seen.add(res_num)
                except ValueError:
                    continue
    return residues


def write_labeled_pdb(pdb_path: str, predictions: np.ndarray, output_path: str):
    """Write PDB with predictions in B-factor column."""
    residue_indices = get_residue_indices(pdb_path)
    label_map = {res_idx: predictions[i] for i, res_idx in enumerate(residue_indices) 
                 if i < len(predictions)}
    
    with open(pdb_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    res_num = int(line[22:26].strip())
                    pred_val = label_map.get(res_num, 0.0)
                    new_line = line[:60] + f'{pred_val:6.2f}' + line[66:]
                    f_out.write(new_line)
                except (ValueError, IndexError):
                    f_out.write(line)
            else:
                f_out.write(line)


def generate_pymol_script(labeled_pdb_path: str, predictions: list, threshold: float, 
                          builder) -> str:
    """Generate PyMOL script to visualize binding predictions with confidence labels."""
    n_binding = sum(1 for p in predictions if p >= threshold)
    n_total = len(predictions)
    
    script = f'''# GraphBind-UWU Prediction Visualization
# Generated automatically

# Load the labeled structure (B-factor = binding probability)
load {labeled_pdb_path}, protein

# Basic settings
bg_color white
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set antialias, 2
set ray_shadows, 0
set label_size, 14
set label_color, black

# Show as cartoon
show cartoon, protein
hide lines

# Color entire protein white
color white, protein

# Select and color binding residues (B-factor >= threshold)
select binding, protein and b > {threshold - 0.001}
color red, binding
delete binding

'''
    
    # Add labels for binding residues with confidence
    for i, (res, pred) in enumerate(zip(builder.residues, predictions)):
        if pred >= threshold:
            chain_id = res.get_parent().id
            resseq = res.get_id()[1]
            resname = res.get_resname()
            chain_sel = f"chain {chain_id} and " if chain_id.strip() else ""
            script += f'label {chain_sel}resi {resseq} and name CA, "{resname}{resseq}: {pred:.0%}"\n'
    
    script += f'''
# Zoom and orient
zoom protein
orient

# Print summary
print "=========================================="
print "GraphBind-UWU Binding Site Prediction"
print "=========================================="
print "  White = Non-binding"
print "  Red   = Binding (with confidence labels)"
print ""
print "Threshold: {threshold}"
print "Binding residues: {n_binding} / {n_total}"
print "=========================================="
'''
    
    return script


def visualize_in_pymol(labeled_pdb_path: str, predictions: list, threshold: float,
                       builder, output_dir: Path):
    """Launch PyMOL with visualization script."""
    script = generate_pymol_script(labeled_pdb_path, predictions, threshold, builder)
    
    script_path = output_dir / "visualize_binding.pml"
    with open(script_path, 'w') as f:
        f.write(script)
    print(f"[Output] PyMOL script saved to: {script_path}")
    
    try:
        subprocess.Popen(['pymol', str(script_path)], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        print("[Visualization] PyMOL launched successfully!")
    except FileNotFoundError:
        print("[Warning] PyMOL not found in PATH. Run manually with:")
        print(f"  pymol {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein binding sites using GraphBind-UWU model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py --pdb protein.pdb
    python predict.py --pdb protein.pdb --threshold 0.6
    python predict.py --pdb protein.pdb --no-pymol --output results/
        """
    )
    
    parser.add_argument("--pdb", "-p", required=True, help="Path to input PDB file")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Binding probability threshold (default: 0.5)")
    parser.add_argument("--checkpoint", "-c", default=None, help="Optional local checkpoint path; otherwise download from Hugging Face")
    parser.add_argument("--hf-repo", default=HF_REPO_ID, help="Hugging Face repository that stores the checkpoint")
    parser.add_argument("--hf-filename", default=HF_CHECKPOINT_FILENAME, help="Checkpoint filename inside the Hugging Face repository")
    parser.add_argument("--hf-cache-dir", default=str(HF_CACHE_DIR), help="Directory used to cache downloaded checkpoints")
    parser.add_argument("--no-pymol", action="store_true", help="Don't launch PyMOL visualization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    pdb_path = Path(args.pdb).resolve()
    if not pdb_path.exists():
        print(f"[Error] PDB file not found: {pdb_path}")
        sys.exit(1)
    
    try:
        checkpoint_path = resolve_checkpoint_path(
            args.checkpoint,
            args.hf_repo,
            args.hf_filename,
            Path(args.hf_cache_dir).expanduser(),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[Error] {exc}")
        sys.exit(1)
    
    # Create temp folder for outputs
    output_dir = Path(tempfile.mkdtemp(prefix=f"graphbind_{pdb_path.stem}_"))
    print(f"[Config] Output directory: {output_dir}")
    
    device = torch.device(args.device)
    print(f"[Config] Using device: {device}")
    print(f"[Config] Threshold: {args.threshold}")
    
    # Load model
    print("\n[Step 1/3] Loading model...")
    model = load_model(str(checkpoint_path), device)
    
    # Preprocess PDB
    print("\n[Step 2/3] Preprocessing PDB...")
    esm_device = args.device if args.device in ["cuda", "mps"] else "cpu"
    data, builder = preprocess_pdb(str(pdb_path), device=esm_device)
    data = data.to(device)
    
    # Run prediction
    print("\n[Step 3/3] Running prediction...")
    with torch.no_grad():
        predictions = model.predict(data).cpu().tolist()  # Use tolist() instead of numpy()
    
    # Results summary
    n_binding = sum(1 for p in predictions if p >= args.threshold)
    n_total = len(predictions)
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Total residues:    {n_total}")
    print(f"Binding residues:  {n_binding} ({100*n_binding/n_total:.1f}%)")
    print(f"Threshold:         {args.threshold}")
    print("="*50)
    
    # Save predictions to CSV
    csv_path = output_dir / f"{pdb_path.stem}_predictions.csv"
    with open(csv_path, 'w') as f:
        f.write("residue_index,chain,residue_number,residue_name,binding_probability,binding_prediction\n")
        for i, (res, pred) in enumerate(zip(builder.residues, predictions)):
            chain_id = res.get_parent().id
            res_id = res.get_id()
            resseq = res_id[1]
            resname = res.get_resname()
            is_binding = int(pred >= args.threshold)
            f.write(f"{i},{chain_id},{resseq},{resname},{pred:.4f},{is_binding}\n")
    print(f"\n[Output] Predictions saved to: {csv_path}")
    
    # Save labeled PDB
    labeled_pdb_path = output_dir / f"{pdb_path.stem}_binding.pdb"
    write_labeled_pdb(str(pdb_path), predictions, str(labeled_pdb_path))
    print(f"[Output] Labeled PDB saved to: {labeled_pdb_path}")
    
    # Visualize in PyMOL
    if not args.no_pymol:
        print("\n[Visualization] Launching PyMOL...")
        visualize_in_pymol(str(labeled_pdb_path), predictions, args.threshold, builder, output_dir)
    
    print("\n[Done] Prediction complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
