import numpy as np
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser, DSSP, Selection
from Bio.Data.PDBData import protein_letters_3to1 as three_to_one
import torch
from torch_geometric.data import Data
import esm
import hashlib
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from ..constants.kyte_doolittle import Hydrophobicity
from ..constants.formal_charge import formal_charge
from ..constants.isoelectric import Isoelectric
from ..constants.side_chain import Sidechain_length




HYDROPHOBICITY = Hydrophobicity
FORMAL_CHARGE = formal_charge
ISOELECTRIC_POINT = Isoelectric
SIDECHAIN_LENGTH = Sidechain_length

STANDARD_AA = set(HYDROPHOBICITY.keys())
DIELECTRIC_CONSTANT = 4.0   # distance-dependent dielectric (protein interior)

VDW_CUTOFF = 8.0
HBOND_ENERGY_THRESHOLD = -0.5
PEPTIDE_BOND_CUTOFF = 1.5  

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# Edge type → integer label (for concatenation later)
EDGE_TYPE_MAP = {"peptide": 0, "vdw": 1, "hbond": 2}


class ProteinGraphBuilder:
    def __init__(self, pdb_path: str):
        self.pdb_path = pdb_path

        # --- Parse structure ---
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", pdb_path)
        self.model = next(self.structure.get_models())

        self._raw_residues = [
            r for r in Selection.unfold_entities(self.model, "R")
            if "CA" in r
        ]

        # --- Sequence extraction + index mapping ---
        self.sequence, self.valid_indices, self.skipped = self._extract_sequence()

        # Final residue list — only standard AA residues, aligned to sequence
        self.residues = [self._raw_residues[i] for i in self.valid_indices]
        self.res_to_idx = {r: i for i, r in enumerate(self.residues)}

        # Precompute once — used by vdw_edges
        self._cb_coords_numpy = self._get_cb_coords()
        self._cb_coords = torch.Tensor(self._cb_coords_numpy)  # (N, 3)

        # Build DSSP → node index map once — used by hbond_edges
        self._dssp_to_node_idx = self._build_dssp_map()



        print(f"[ProteinGraphBuilder] {len(self.residues)} residues | "
              f"sequence length: {len(self.sequence)}")
        if self.skipped:
            print(f"  Skipped: {[(i, r.get_resname(), reason) for i, r, reason in self.skipped]}")

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    def _extract_sequence(self) -> tuple[str, list[int], list]:
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
            if icode.strip():
                print(f"  [insertion] resseq={resseq} icode={icode} ({resname}) at index {i}")

            sequence.append(one_letter)
            valid_indices.append(i)

        return "".join(sequence), valid_indices, skipped

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_cb_coords(self) -> np.ndarray:
        """
        Cβ for all standard residues, Cα fallback for Glycine.
        Precomputed once at init — do not call this repeatedly.
        """
        coords = []
        for res in self.residues:
            atom = res["CB"] if "CB" in res else res["CA"]
            coords.append(atom.get_vector().get_array())
        return np.array(coords)  # (N, 3)

    # ------------------------------------------------------------------
    # DSSP → node index mapping
    # ------------------------------------------------------------------

    def _build_dssp_map(self) -> dict:
        """
        Maps each DSSP key (chain_id, res_id) → node index in self.residues.
        DSSP may drop terminal residues or residues it can't assign, so this
        mapping will be partial — missing entries are silently skipped in hbond_edges.
        """
        dssp_map = {}
        for res in self.residues:
            chain_id = res.get_parent().id
            res_id = res.get_id()          # (het, resseq, icode)
            dssp_key = (chain_id, res_id)
            node_idx = self.res_to_idx[res]
            dssp_map[dssp_key] = node_idx
        return dssp_map

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------

    def peptide_edges(self) -> list[tuple]:
        """
        Consecutive residues in the same chain connected by a real C→N bond.
        Checks atomic distance to exclude chain breaks (missing residues in PDB).
        Weight is fixed at 1.0 — covalent bonds are binary.
        """
        edges = []
        for i in range(len(self.residues) - 1):
            r1, r2 = self.residues[i], self.residues[i + 1]

            # Must be same chain
            if r1.get_parent().id != r2.get_parent().id:
                continue

            # Check actual C(i) → N(i+1) bond distance to detect chain breaks
            if "C" not in r1 or "N" not in r2:
                continue
            c_coord = r1["C"].get_vector().get_array()
            n_coord = r2["N"].get_vector().get_array()
            bond_dist = float(np.linalg.norm(c_coord - n_coord))

            if bond_dist <= PEPTIDE_BOND_CUTOFF:
                edges.append((i, i + 1, 1.0, "peptide"))

        return edges

    def vdw_edges(self) -> list[tuple]:

        tree = cKDTree(self._cb_coords_numpy)
        edges = []
        for i, j in tree.query_pairs(r=VDW_CUTOFF):
            dist = float(np.linalg.norm(self._cb_coords[i] - self._cb_coords[j]))
            weight = float(np.exp(-dist / VDW_CUTOFF))
            edges.append((i, j, weight, "vdw"))
        return edges

    def hbond_edges(self) -> list[tuple]:
        dssp = DSSP(self.model, self.pdb_path, dssp="mkdssp", file_type="PDB")
        edges = []
 
        dssp_keys_list = list(dssp.keys())
 
        for idx, key in enumerate(dssp_keys_list):
            data = dssp[key]
            for offset_field, energy_field in [(6, 7), (8, 9)]:
                offset = data[offset_field]
                energy = data[energy_field]
 
                if energy >= HBOND_ENERGY_THRESHOLD or offset == 0:
                    continue
 
                partner_dssp_idx = idx + offset
                if not (0 <= partner_dssp_idx < len(dssp_keys_list)):
                    continue
 
                partner_key = dssp_keys_list[partner_dssp_idx]
                src_node = self._dssp_to_node_idx.get(key)
                dst_node = self._dssp_to_node_idx.get(partner_key)
 
                if src_node is None or dst_node is None:
                    continue
 
                edges.append((src_node, dst_node, float(abs(energy)), "hbond"))
 
        return edges
    
    def get_coulomb_term(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:

        charges = torch.tensor(
            [float(FORMAL_CHARGE.get(r.get_resname().strip(), 0)) for r in self.residues],
            dtype=torch.float
        )
    
        q_src = charges[src]   # (E,)
        q_dst = charges[dst]   # (E,)
    
        src_np = self._cb_coords_numpy[src.numpy()]
        dst_np = self._cb_coords_numpy[dst.numpy()]
        r = torch.tensor(np.linalg.norm(src_np - dst_np, axis=1),dtype=torch.float).clamp(min=1e-6)  # avoid division by zero
    
        coulomb = q_src * q_dst / (DIELECTRIC_CONSTANT * r ** 2)
        return coulomb  # (E,)
    

    ### ----------------------------------------------------------------
    # Node feature builders
    # ----------------------------------------------------------------

    def get_one_hot(self) -> torch.Tensor:

        aa_list = sorted(STANDARD_AA)
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    
        one_hot = torch.zeros(len(self.residues), len(aa_list))
        for i, res in enumerate(self.residues):
            resname = res.get_resname().strip()
            if resname in aa_to_idx:
                one_hot[i, aa_to_idx[resname]] = 1.0
        return one_hot
    

    def get_hydrophobicity(self) -> torch.Tensor:
        """
        Kyte-Doolittle scale. Returns (N,) float tensor.
        Range: -4.5 (hydrophilic) to +4.5 (hydrophobic).
        """

        return torch.tensor(
            [[HYDROPHOBICITY.get(r.get_resname().strip(), 0.0)] for r in self.residues],
            dtype=torch.float)
    
    def get_formal_charge(self) -> torch.Tensor:
        """
        Integer formal charge at pH 7. Returns (N,) float tensor.
        Values: -1, 0, or +1.
        """
        return torch.tensor(
            [[float(FORMAL_CHARGE.get(r.get_resname().strip(), 0))] for r in self.residues],
            dtype=torch.float
        )

        
        
    def get_isoelectric_point(self) -> torch.Tensor:
        """
        Approximate pI per residue. Returns (N,) float tensor.
        """
        return torch.tensor(
            [[ISOELECTRIC_POINT.get(r.get_resname().strip(), 5.97)] for r in self.residues],
            dtype=torch.float
        )
    
    def get_sidechain_length(self, use_coords: bool = True) -> torch.Tensor:
        """
        Side chain length per residue.
    
        Args:
            use_coords: If True, computes Euclidean distance between Cα and
                        the farthest sidechain heavy atom from PDB coordinates.
                        More physically accurate but requires complete sidechain.
                        Falls back to lookup table if sidechain atoms are missing.
                        If False, uses the bond-count lookup table directly.
    
        Returns: (N,) float tensor
        """
        lengths = []
        for res in self.residues:
            resname = res.get_resname().strip()
    
            if use_coords and "CA" in res:
                ca = res["CA"].get_vector().get_array()
                sidechain_atoms = [
                    atom for atom in res.get_atoms()
                    if atom.get_name() not in ("N", "CA", "C", "O")  # exclude backbone
                ]
                if sidechain_atoms:
                    distances = [
                        np.linalg.norm(atom.get_vector().get_array() - ca)
                        for atom in sidechain_atoms
                    ]
                    lengths.append([float(max(distances))])
                    continue
    
            # Fallback to bond count lookup
            lengths.append([float(SIDECHAIN_LENGTH.get(resname, 0))])
    
        return torch.tensor(lengths, dtype=torch.float)
    
    def b_factor(self) -> torch.Tensor:
        """
        B-factor (temperature factor) per residue, averaged over all atoms.
        Returns (N,) float tensor.
        """
        b_factors = []
        for res in self.residues:
            atoms = list(res.get_atoms())
            if atoms:
                avg_b = np.mean([atom.get_bfactor() for atom in atoms])
                b_factors.append([float(avg_b)])
            else:
                b_factors.append([0.0])  # default if no atoms
        return torch.tensor(b_factors, dtype=torch.float)

    # ------------------------------------------------------------------
    # Graph builder
    # ------------------------------------------------------------------

    def build(self, node_features: torch.Tensor | None = None,
            contacts: torch.Tensor | None = None, y: torch.Tensor | None = None) -> Data:

        all_edges = self.peptide_edges() + self.vdw_edges() + self.hbond_edges()

        src = torch.tensor([e[0] for e in all_edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in all_edges], dtype=torch.long)

        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

        weights = torch.tensor([e[2] for e in all_edges], dtype=torch.float)
        types   = torch.tensor([EDGE_TYPE_MAP[e[3]] for e in all_edges], dtype=torch.float)

        zero        = torch.zeros_like(weights)
        contacts_fw = contacts[src, dst] if contacts is not None else zero
        contacts_rv = contacts[dst, src] if contacts is not None else zero

        coulomb_fw = self.get_coulomb_term(src, dst)
        coulomb_rv = self.get_coulomb_term(dst, src)

        scalar_fw = torch.stack([weights, types, contacts_fw, coulomb_fw], dim=1)  # (E, 4)
        scalar_rv = torch.stack([weights, types, contacts_rv, coulomb_rv], dim=1)  # (E, 4)
        edge_attr = torch.cat([scalar_fw, scalar_rv], dim=0)                       # (2E, 4)

        # None check before concat — zeros must match ESM2 dim so cat stays (N, 1304)
        if node_features is None:
            print("[ProteinGraphBuilder] No node features — using zeros.")
            node_features = torch.zeros((len(self.residues), 1280), dtype=torch.float)  # ← 1280 not 1

        assert node_features.shape[0] == len(self.residues), (
            f"node_features {node_features.shape[0]} != residues {len(self.residues)}"
        )

        node_features = torch.cat([
            node_features,                               # (N, 1280)
            self.get_one_hot(),                          # (N, 20)
            self.get_hydrophobicity(),                   # (N, 1)
            self.get_formal_charge(),                    # (N, 1)
            self.get_isoelectric_point(),                # (N, 1)
            self.get_sidechain_length(use_coords=True),  # (N, 1)
            self.b_factor(),                            # (N, 1)
        ], dim=1)  # (N, 1304)

        return Data(
            x=node_features,
            pos=self._cb_coords,        # torch.Tensor (N, 3)
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.residues),
            y=y

        )
        

@dataclass
class ESMOutput:
    embeddings: torch.Tensor   # (L, 1280) — per-residue, last layer
    contacts:   torch.Tensor   # (L, L)    — predicted from attention heads
    sequence:   str


class ESMProcessor:
    def __init__(
        self,
        cache_dir: str = ".esm_cache",
        device: str = "cuda",
        repr_layer: int = 33,          # last layer of esm2_t33_650M_UR50D
        batch_token_limit: int = 1024,
    ):
        self.device = device
        self.repr_layer = repr_layer
        self.batch_token_limit = batch_token_limit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            print("[ESMProcessor] Loading ESM2...")
            self._model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._batch_converter = self._alphabet.get_batch_converter()
            self._model = self._model.eval().to(self.device)
            print("[ESMProcessor] Ready.")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_path(self, sequence: str) -> Path:
        key = hashlib.md5(sequence.encode()).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _load_cache(self, sequence: str) -> ESMOutput | None:
        path = self._cache_path(sequence)
        if path.exists():
            data = torch.load(path, map_location="cpu")
            return ESMOutput(**data)
        return None

    def _save_cache(self, sequence: str, output: ESMOutput):
        torch.save({
            "embeddings": output.embeddings,
            "contacts":   output.contacts,
            "sequence":   output.sequence,
        }, self._cache_path(sequence))

    # ------------------------------------------------------------------
    # Core inference — takes (label, seq) tuples, returns raw results
    # ------------------------------------------------------------------

    def _run_inference(self, labeled_seqs: list[tuple[str, str]]) -> dict:
        """
        labeled_seqs: list of ("label", "SEQUENCE") tuples
        Returns raw ESM2 results dict.
        """
        _, _, batch_tokens = self._batch_converter(labeled_seqs)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self._model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=True,
            )
        return results, batch_tokens

    # ------------------------------------------------------------------
    # Single sequence
    # ------------------------------------------------------------------

    def process(self, sequence: str) -> ESMOutput:
        cached = self._load_cache(sequence)
        if cached is not None:
            return cached

        self._load_model()

        results, _ = self._run_inference([("protein", sequence)])

        L = len(sequence)
        output = ESMOutput(
            # Slice [1 : L+1] to remove BOS token (index 0) and EOS token (index L+1)
            embeddings=results["representations"][self.repr_layer][0, 1:L+1].cpu(),  # (L, 1280)
            contacts=results["contacts"][0, :L, :L].cpu(),                           # (L, L)
            sequence=sequence,
        )
        self._save_cache(sequence, output)
        return output

    # ------------------------------------------------------------------
    # Batch — length sorted to minimize padding waste
    # ------------------------------------------------------------------

    def process_batch(self, sequences: list[str]) -> list[ESMOutput]:
        results = {}

        # Cache hits
        uncached = []
        for i, seq in enumerate(sequences):
            cached = self._load_cache(seq)
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, seq))

        if not uncached:
            return [results[i] for i in range(len(sequences))]

        self._load_model()

        # Sort by length — minimizes padding waste per bucket
        uncached.sort(key=lambda x: len(x[1]))

        # Build length-aware buckets within token limit
        batches = []
        current_batch = []
        current_max_len = 0

        for i, seq in uncached:
            new_max = max(current_max_len, len(seq))
            projected_tokens = new_max * (len(current_batch) + 1)

            if current_batch and projected_tokens > self.batch_token_limit:
                batches.append(current_batch)
                current_batch = [(i, seq)]
                current_max_len = len(seq)
            else:
                current_batch.append((i, seq))
                current_max_len = new_max

        if current_batch:
            batches.append(current_batch)

        # Run inference per bucket
        for batch in batches:
            indices, seqs = zip(*batch)
            labeled = [(str(idx), seq) for idx, seq in zip(indices, seqs)]

            raw, _ = self._run_inference(labeled)

            for k, (orig_idx, seq) in enumerate(zip(indices, seqs)):
                L = len(seq)
                output = ESMOutput(
                    embeddings=raw["representations"][self.repr_layer][k, 1:L+1].cpu(),
                    contacts=raw["contacts"][k, :L, :L].cpu(),
                    sequence=seq,
                )
                self._save_cache(seq, output)
                results[orig_idx] = output

        return [results[i] for i in range(len(sequences))]
    

def parse_binding_residues(binding_str: str) -> list[tuple[str, int]]:
    """
    Parses 'F43 R45 V68 S92' → [('F', 43), ('R', 45), ('V', 68), ('S', 92)]
    Returns list of (one_letter_AA, resseq) tuples.
    """
    result = []
    for token in binding_str.split():
        aa      = token[0]          # one-letter AA — use for validation
        resseq  = int(token[1:])    # PDB residue number
        result.append((aa, resseq))
    return result


def get_binding_indices(
    builder: "ProteinGraphBuilder",
    binding_residues: list[tuple[str, int]],
    validate_aa: bool = True,
) -> list[int]:
    """
    Maps (AA, resseq) tuples → node indices in builder.residues.
    Skips residues not found in the structure (chain breaks, filtered out, etc).

    Args:
        validate_aa: If True, warns when the AA letter doesn't match the
                     residue at that resseq — catches PDB numbering mismatches.
    """
    from Bio.Data.PDBData import protein_letters_3to1 as three_to_one

    # Build resseq → (node_idx, one_letter) lookup
    resseq_map = {}
    for i, res in enumerate(builder.residues):
        resseq  = res.get_id()[1]                          # integer PDB resseq
        resname = res.get_resname().strip()
        one_letter = three_to_one.get(resname, "?")
        resseq_map[resseq] = (i, one_letter)

    binding_indices = []
    for aa, resseq in binding_residues:
        if resseq not in resseq_map:
            print(f"  [warning] resseq {resseq} ({aa}) not found in structure — skipped")
            continue

        node_idx, actual_aa = resseq_map[resseq]

        if validate_aa and aa != actual_aa:
            print(f"  [warning] resseq {resseq}: expected {aa}, found {actual_aa} — included anyway")

        binding_indices.append(node_idx)

    return binding_indices
    


class ProteinDataset(Dataset):
    def __init__(self, pdb_paths: list[str], binding_residues: list[str], esm_processor: ESMProcessor):
        self.pdb_paths = pdb_paths
        self.processor = esm_processor

        # Build all graph builders first (fast, CPU only)
        print("Parsing PDB files...")
        self.builders = [ProteinGraphBuilder(p) for p in pdb_paths]

        print("Processing binding residues...")

        self.binding_residues = [parse_binding_residues(br) for br in binding_residues]
        self.binding_residue_indices = [
            get_binding_indices(builder, br, validate_aa=True)
            for builder, br in zip(self.builders, self.binding_residues)
        ]

        # Batch all sequences through ESMFold in one pass
        print("Running ESMFold...")
        sequences = [b.sequence for b in self.builders]
        self.esm_outputs = self.processor.process_batch(sequences)


    def __len__(self):
        return len(self.builders)

    def __getitem__(self, idx) -> Data:
        builder = self.builders[idx]
        esm_out = self.esm_outputs[idx]
        bind_idx   = self.binding_residue_indices[idx]

        y = torch.zeros(len(builder.residues), dtype=torch.float)
        y[bind_idx] = 1.0


        # ESMFold also gives you a contact map — you can add it as edge weights
        # or use it to add/filter edges here before building
        node_features = esm_out.embeddings   # (L, 1024) — projection happens in GNN
        edge_attention = esm_out.contacts
        return builder.build(node_features=node_features, contacts = edge_attention, y=y)
    

 