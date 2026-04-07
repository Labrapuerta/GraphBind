from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB import PDBList, PDBIO, Select


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id


def download_and_extract_chain(pdb_id: str, chain: str, save_dir: str) -> str:
    """
    Downloads full PDB via PDBList then extracts the specified chain.
    """
    save_dir = Path(save_dir)
    out_path = save_dir / f"{pdb_id}{chain}.pdb"

    if out_path.exists():
        return str(out_path)

    # Download full structure
    pdbl = PDBList()
    full_path = pdbl.retrieve_pdb_file(pdb_id, pdir=str(save_dir), file_type="pdb")

    # Extract chain
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, full_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect(chain))

    # Remove the full PDB — no longer needed
    Path(full_path).unlink()
    print(f"Extracted chain {chain} from {pdb_id} → {out_path.name}")

    return str(out_path)


def parse_biolip(annotation_file: str, receptor_dir: str) -> tuple[list[str], list[str]]:
    pdb_paths, binding_residues = [], []

    with open(annotation_file) as f:
        for line in f:
            cols = line.strip().split("\t")
            pdb_id, chain, binding = cols[0], cols[1], cols[7]

            if binding == "-" or not binding.strip():
                continue

            # Option 1 — BioLiP direct (recommended)
            path = download_and_extract_chain(pdb_id, chain, receptor_dir)

            # Option 2 — PDBList + chain extraction
            # path = download_and_extract_chain(pdb_id, chain, receptor_dir)

            if path is None:
                continue

            pdb_paths.append(path)
            binding_residues.append(binding.strip())

    return pdb_paths, binding_residues