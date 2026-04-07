"""
preprocess.py
=============
Optimized preprocessing pipeline with streaming batch processing.

Features:
    - Processes proteins in batches of N (default: 10) to avoid OOM
    - Deletes train PDB files after processing, keeps test PDB files
    - Saves each graph with torch.save into CV-specific folders
    - Returns a manifest CSV with all paths for lazy loading

Output structure:
    preprocessed/
    ├── train/
    │   ├── cv_1/   (.pt files for CV fold 1)
    │   ├── cv_2/
    │   ├── cv_3/
    │   ├── cv_4/
    │   └── cv_5/
    ├── test/       (.pt files for test set)
    ├── pdb_test/   (kept PDB files for test set)
    └── manifest.csv

Usage:
    python -m src.preprocessing.preprocess \\
        --csv       pdb_splits_CL.csv \\
        --out_dir   preprocessed \\
        --batch_size 10 \\
        --device    cuda
"""

import argparse
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from Bio.PDB import PDBParser, PDBIO, Select, PDBList

# Project imports
from .GraphBuilder import (
    ProteinGraphBuilder,
    ESMProcessor,
    parse_binding_residues,
    get_binding_indices,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single protein."""
    pdb_id: str
    chain: str
    split: str
    cv_batch: int
    pt_path: Optional[str] = None
    pdb_path: Optional[str] = None
    n_residues: int = 0
    n_binding: int = 0
    n_edges: int = 0
    status: str = "pending"
    error: Optional[str] = None


# =============================================================================
# PDB Download and Chain Extraction
# =============================================================================

class ChainSelect(Select):
    """Keeps only ATOM records for the requested chain — no HETATM, no ligands."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.id == self.chain_id

    def accept_residue(self, residue):
        het, resseq, icode = residue.get_id()
        return het.strip() == ""


def download_pdb(pdb_id: str, raw_dir: Path) -> Optional[Path]:
    """Downloads a PDB using BioPython's PDBList. Returns saved path or None."""
    dest = raw_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest

    try:
        pdbl = PDBList(verbose=False)
        # Download in PDB format (saves as pdb{id}.ent)
        pdbl.download_pdb_files([pdb_id], pdir=str(raw_dir), file_format="pdb")

        # PDBList saves as pdb{id}.ent, rename to {id}.pdb
        ent_path = raw_dir / f"pdb{pdb_id.lower()}.ent"
        if not ent_path.exists() or ent_path.stat().st_size < 100:
            ent_path.unlink(missing_ok=True)
            return None

        ent_path.rename(dest)
        return dest
    except Exception:
        dest.unlink(missing_ok=True)
        return None


def _extract_cryst1_header(pdb_path: Path) -> str:
    """Extract CRYST1 line from a PDB file, or return a dummy one for DSSP compatibility."""
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("CRYST1"):
                    return line.rstrip('\n')
    except Exception:
        pass
    # Default CRYST1 line (P1 space group, 1x1x1 unit cell) - DSSP just needs this to exist
    return "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1"


def download_and_extract_chain(
    pdb_id: str,
    chain: str,
    raw_dir: Path,
    out_path: Path,
) -> Optional[Path]:
    """
    Downloads the full PDB from RCSB using PDBList, extracts ATOM-only chain records,
    preserves CRYST1 header (required by DSSP), saves to out_path, then deletes the full PDB.
    """
    if out_path.exists():
        return out_path

    full_path = download_pdb(pdb_id, raw_dir)
    if full_path is None:
        return None

    try:
        # Extract CRYST1 header before parsing (PDBIO doesn't preserve it)
        cryst1_line = _extract_cryst1_header(full_path)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(full_path))
        io = PDBIO()
        io.set_structure(structure)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save chain to a temporary string
        from io import StringIO
        temp_output = StringIO()
        io.save(temp_output, ChainSelect(chain))
        chain_content = temp_output.getvalue()

        # Write output with CRYST1 header preserved (required by DSSP)
        with open(out_path, 'w') as f:
            f.write(cryst1_line + '\n')
            f.write(chain_content)

        full_path.unlink(missing_ok=True)
        return out_path
    except Exception:
        full_path.unlink(missing_ok=True)
        return None


# =============================================================================
# Batch Processor (Memory-Efficient)
# =============================================================================

class StreamingBatchProcessor:
    """
    Processes proteins in small batches to avoid memory issues.

    Key optimization: Instead of loading all ESM embeddings into memory,
    we process batch_size proteins at a time, save them, then free memory.
    """

    def __init__(
        self,
        esm_processor: ESMProcessor,
        batch_size: int = 10,
        out_dir: Path = Path("preprocessed"),
        keep_test_pdb: bool = True,
    ):
        self.esm = esm_processor
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.keep_test_pdb = keep_test_pdb

        # Create output directories
        self.train_dir = out_dir / "train"
        self.test_dir = out_dir / "test"
        self.test_pdb_dir = out_dir / "pdb_test"

        for cv in range(1, 6):
            (self.train_dir / f"cv_{cv}").mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.test_pdb_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_paths(self, row: pd.Series) -> tuple[Path, Optional[Path]]:
        """Returns (pt_path, pdb_path) for a given row."""
        pdb_id = row["PDB_ID"]
        chain = row["Chains_To_Keep"]
        label = f"{pdb_id}_{chain}"

        if row["Split"] == "training":
            cv = int(row["CV_Batch"])
            pt_path = self.train_dir / f"cv_{cv}" / f"{label}.pt"
            pdb_path = None  # Train PDBs will be deleted
        else:
            pt_path = self.test_dir / f"{label}.pt"
            pdb_path = self.test_pdb_dir / f"{label}.pdb" if self.keep_test_pdb else None

        return pt_path, pdb_path

    def process_single(
        self,
        row: pd.Series,
        pdb_path: Path,
        esm_output: "ESMOutput",
    ) -> ProcessingResult:
        """Process a single protein and save the graph."""
        result = ProcessingResult(
            pdb_id=row["PDB_ID"],
            chain=row["Chains_To_Keep"],
            split=row["Split"],
            cv_batch=int(row["CV_Batch"]) if pd.notna(row["CV_Batch"]) else 0,
        )

        pt_path, final_pdb_path = self._get_output_paths(row)

        # Skip if already processed
        if pt_path.exists():
            result.pt_path = str(pt_path)
            result.status = "ok_cached"
            # Clean up PDB if train
            if row["Split"] == "training":
                pdb_path.unlink(missing_ok=True)
            return result

        try:
            # Parse structure
            builder = ProteinGraphBuilder(str(pdb_path))
            parsed = parse_binding_residues(row["Binding_Residues"])
            bind_idxs = get_binding_indices(builder, parsed, validate_aa=True)

            # Create label tensor
            y = torch.zeros(len(builder.residues), dtype=torch.float)
            if bind_idxs:
                y[list(bind_idxs)] = 1.0

            # Build graph
            data = builder.build(
                node_features=esm_output.embeddings,
                contacts=esm_output.contacts,
                y=y,
            )

            # Save graph
            torch.save(data, pt_path)

            # Populate result
            result.pt_path = str(pt_path)
            result.n_residues = len(builder.residues)
            result.n_binding = int(y.sum())
            result.n_edges = data.edge_index.shape[1]
            result.status = "ok"

            # Handle PDB file
            if final_pdb_path is not None:
                # Copy to test PDB folder
                import shutil
                shutil.copy2(pdb_path, final_pdb_path)
                result.pdb_path = str(final_pdb_path)

            # Delete train PDB files
            if row["Split"] == "training":
                pdb_path.unlink(missing_ok=True)

        except Exception as e:
            result.status = "fail"
            result.error = str(e)
            traceback.print_exc()

        return result

    def process_batch(
        self,
        batch_df: pd.DataFrame,
        pdb_paths: dict[int, Path],
        batch_num: int,
        total_batches: int,
    ) -> list[ProcessingResult]:
        """
        Process a batch of proteins:
        1. Collect sequences for valid PDBs
        2. Run ESM2 on the batch
        3. Build and save graphs
        4. Free memory
        """
        results = []

        # Collect valid entries for ESM processing
        valid_entries = []
        for idx, row in batch_df.iterrows():
            pdb_path = pdb_paths.get(idx)
            if pdb_path is None or not Path(pdb_path).exists():
                results.append(ProcessingResult(
                    pdb_id=row["PDB_ID"],
                    chain=row["Chains_To_Keep"],
                    split=row["Split"],
                    cv_batch=int(row["CV_Batch"]) if pd.notna(row["CV_Batch"]) else 0,
                    status="fail",
                    error="PDB download failed",
                ))
                continue

            try:
                builder = ProteinGraphBuilder(str(pdb_path))
                valid_entries.append({
                    "idx": idx,
                    "row": row,
                    "pdb_path": Path(pdb_path),
                    "sequence": builder.sequence,
                    "builder": builder,
                })
            except Exception as e:
                results.append(ProcessingResult(
                    pdb_id=row["PDB_ID"],
                    chain=row["Chains_To_Keep"],
                    split=row["Split"],
                    cv_batch=int(row["CV_Batch"]) if pd.notna(row["CV_Batch"]) else 0,
                    status="fail",
                    error=f"Parse failed: {e}",
                ))

        if not valid_entries:
            return results

        # Run ESM2 on batch
        print(f"  [batch {batch_num}/{total_batches}] Running ESM2 on {len(valid_entries)} sequences...")
        sequences = [e["sequence"] for e in valid_entries]

        try:
            esm_outputs = self.esm.process_batch(sequences)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and isinstance(e, RuntimeError):
                raise
            print(f"  [ESM OOM] Batch failed, retrying one-by-one...")
            gc.collect()
            torch.cuda.empty_cache()

            esm_outputs = []
            failed_indices = []
            for k, entry in enumerate(valid_entries):
                try:
                    out = self.esm.process(entry["sequence"])
                    esm_outputs.append(out)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e2:
                    if "out of memory" not in str(e2).lower() and isinstance(e2, RuntimeError):
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    esm_outputs.append(None)
                    failed_indices.append(k)

            # Record individual failures 
            for k in sorted(failed_indices, reverse=True):
                entry = valid_entries[k]
                results.append(ProcessingResult(
                    pdb_id=entry["row"]["PDB_ID"],
                    chain=entry["row"]["Chains_To_Keep"],
                    split=entry["row"]["Split"],
                    cv_batch=int(entry["row"]["CV_Batch"]) if pd.notna(entry["row"]["CV_Batch"]) else 0,
                    status="fail",
                    error=f"ESM OOM: sequence too long ({len(entry['sequence'])} residues)",
                ))
                valid_entries.pop(k)
                esm_outputs.pop(k)

            if not valid_entries:
                return results
        except Exception as e:
            print(f"  [ESM error] {e}")
            for entry in valid_entries:
                results.append(ProcessingResult(
                    pdb_id=entry["row"]["PDB_ID"],
                    chain=entry["row"]["Chains_To_Keep"],
                    split=entry["row"]["Split"],
                    cv_batch=int(entry["row"]["CV_Batch"]) if pd.notna(entry["row"]["CV_Batch"]) else 0,
                    status="fail",
                    error=f"ESM failed: {e}",
                ))
            return results

        # Process each protein in the batch
        for entry, esm_out in zip(valid_entries, esm_outputs):
            result = self._process_with_esm(entry, esm_out)
            results.append(result)

            label = f"{result.pdb_id}_{result.chain}"
            if result.status.startswith("ok"):
                print(f"    [saved] {label}.pt ({result.n_residues} res, {result.n_binding} binding)")
            else:
                print(f"    [fail] {label}: {result.error}")

        # Free memory
        del esm_outputs
        for entry in valid_entries:
            del entry["builder"]
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _process_with_esm(self, entry: dict, esm_output) -> ProcessingResult:
        """Process a single entry that already has ESM output."""
        row = entry["row"]
        pdb_path = entry["pdb_path"]

        result = ProcessingResult(
            pdb_id=row["PDB_ID"],
            chain=row["Chains_To_Keep"],
            split=row["Split"],
            cv_batch=int(row["CV_Batch"]) if pd.notna(row["CV_Batch"]) else 0,
        )

        pt_path, final_pdb_path = self._get_output_paths(row)

        # Skip if already processed
        if pt_path.exists():
            result.pt_path = str(pt_path)
            result.status = "ok_cached"
            if row["Split"] == "training":
                pdb_path.unlink(missing_ok=True)
            return result

        try:
            builder = ProteinGraphBuilder(str(pdb_path))
            parsed = parse_binding_residues(row["Binding_Residues"])
            bind_idxs = get_binding_indices(builder, parsed, validate_aa=True)

            # Create label tensor
            y = torch.zeros(len(builder.residues), dtype=torch.float)
            if bind_idxs:
                y[list(bind_idxs)] = 1.0

            # Build graph
            data = builder.build(
                node_features=esm_output.embeddings,
                contacts=esm_output.contacts,
                y=y,
            )

            # Save graph
            torch.save(data, pt_path)

            # Populate result
            result.pt_path = str(pt_path)
            result.n_residues = len(builder.residues)
            result.n_binding = int(y.sum())
            result.n_edges = data.edge_index.shape[1]
            result.status = "ok"

            # Handle PDB file
            if final_pdb_path is not None:
                import shutil
                shutil.copy2(pdb_path, final_pdb_path)
                result.pdb_path = str(final_pdb_path)

            # Delete train PDB files
            if row["Split"] == "training":
                pdb_path.unlink(missing_ok=True)

        except Exception as e:
            result.status = "fail"
            result.error = str(e)

        return result


# =============================================================================
# Main Pipeline
# =============================================================================

def preprocess(
    csv_path: str,
    out_dir: str = "preprocessed",
    raw_dir: str = "data/raw_pdbs",
    esm_cache: str = ".esm_cache",
    batch_size: int = 10,
    device: str = "cuda",
    max_workers: int = 32,
    esm_processor: Optional[ESMProcessor] = None,
) -> pd.DataFrame:
    """
    Main preprocessing pipeline.

    Args:
        csv_path: Path to input CSV with columns:
            - PDB_ID, Chains_To_Keep, Binding_Residues, CV_Batch, Split
        out_dir: Output directory for .pt files
        raw_dir: Temporary directory for raw PDB downloads
        esm_cache: Cache directory for ESM2 model
        batch_size: Number of proteins to process per batch (for ESM)
        device: 'cuda' or 'cpu'
        max_workers: Number of parallel download workers
        esm_processor: Optional ESMProcessor instance. If None, creates a new one.

    Returns:
        DataFrame with manifest (paths to all processed files)
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    chain_dir = raw_dir / "_chains"
    chain_dir.mkdir(exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    print(f"  training : {(df['Split']=='training').sum()}")
    print(f"  testing  : {(df['Split']=='testing').sum()}\n")

    # =========================================================================
    # Step 1: Download and extract chains (parallel)
    # =========================================================================
    print("=" * 60)
    print("Step 1 — Downloading and extracting chains")
    print("=" * 60)

    pdb_paths = {}
    n_done = 0
    total = len(df)

    def _download_row(idx_row):
        idx, row = idx_row
        out_path = chain_dir / f"{row['PDB_ID']}_{row['Chains_To_Keep']}.pdb"
        path = download_and_extract_chain(
            row["PDB_ID"], row["Chains_To_Keep"], raw_dir, out_path
        )
        return idx, row["PDB_ID"], row["Chains_To_Keep"], path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_row, (idx, row)): idx
                   for idx, row in df.iterrows()}
        for future in as_completed(futures):
            idx, pdb_id, chain, path = future.result()
            pdb_paths[idx] = path
            n_done += 1
            status = "ok" if path is not None else "fail"
            print(f"  [{status}] {pdb_id}_{chain}.pdb  ({n_done}/{total})")

    n_ok = sum(1 for p in pdb_paths.values() if p is not None)
    n_fail = total - n_ok
    print(f"\n  Downloaded: {n_ok}  |  Failed: {n_fail}")

    # =========================================================================
    # Step 2: Process in batches
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"Step 2 — Processing in batches of {batch_size}")
    print("=" * 60)

    # Use provided processor or create new one
    if esm_processor is None:
        esm_processor = ESMProcessor(cache_dir=esm_cache, device=device)
    
    processor = StreamingBatchProcessor(
        esm_processor=esm_processor,
        batch_size=batch_size,
        out_dir=out_dir,
        keep_test_pdb=True,
    )

    all_results = []
    indices = list(df.index)
    total_batches = (len(indices) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(indices), batch_size), 1):
        batch_indices = indices[i:i + batch_size]
        batch_df = df.loc[batch_indices]

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch_df)} proteins)")

        results = processor.process_batch(
            batch_df=batch_df,
            pdb_paths={idx: pdb_paths.get(idx) for idx in batch_indices},
            batch_num=batch_num,
            total_batches=total_batches,
        )
        all_results.extend(results)

        # Memory cleanup between batches
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: Create manifest CSV
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3 — Saving manifest")
    print("=" * 60)

    manifest_rows = []
    for result in all_results:
        manifest_rows.append({
            "pdb_id": result.pdb_id,
            "chain": result.chain,
            "split": result.split,
            "cv_batch": result.cv_batch,
            "pt_path": result.pt_path,
            "pdb_path": result.pdb_path,
            "n_residues": result.n_residues,
            "n_binding": result.n_binding,
            "n_edges": result.n_edges,
            "status": result.status,
            "error": result.error,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    ok_count = manifest_df["status"].str.startswith("ok").sum()
    fail_count = len(manifest_df) - ok_count

    print(f"  Total proteins     : {len(df)}")
    print(f"  Successfully saved : {ok_count}")
    print(f"  Failed             : {fail_count}")
    print(f"  Manifest saved to  : {manifest_path}")

    # Show folder structure
    print("\nOutput folder structure:")
    for folder in sorted(out_dir.glob("**/")):
        pt_files = list(folder.glob("*.pt"))
        pdb_files = list(folder.glob("*.pdb"))
        if pt_files or pdb_files:
            size_mb = sum(f.stat().st_size for f in pt_files + pdb_files) / 1e6
            print(f"  {folder.relative_to(out_dir)}/  ->  "
                  f"{len(pt_files)} .pt, {len(pdb_files)} .pdb, {size_mb:.1f} MB")

    return manifest_df


# =============================================================================
# Lazy Dataset Loader (for training)
# =============================================================================

class LazyGraphDataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset that reads .pt files on demand.

    Use this in your training loop to avoid loading all graphs into memory.

    Example:
        manifest = pd.read_csv("preprocessed/manifest.csv")
        train_manifest = manifest[
            (manifest["split"] == "training") &
            (manifest["cv_batch"] != 1)  # Exclude fold 1 for validation
        ]
        dataset = LazyGraphDataset(train_manifest["pt_path"].tolist())
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(self, pt_paths: list[str]):
        self.paths = [p for p in pt_paths if p is not None and Path(p).exists()]
        print(f"LazyGraphDataset: {len(self.paths)} graphs loaded")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return torch.load(self.paths[idx], weights_only=False)


def get_cv_datasets(
    manifest_path: str,
    cv_fold: int,
) -> tuple["LazyGraphDataset", "LazyGraphDataset", "LazyGraphDataset"]:
    """
    Create train/val/test datasets for a specific CV fold.

    Args:
        manifest_path: Path to manifest.csv
        cv_fold: Which fold to use for validation (1-5)

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    manifest = pd.read_csv(manifest_path)
    ok_mask = manifest["status"].str.startswith("ok")
    manifest = manifest[ok_mask]

    # Training: all training folds except cv_fold
    train_mask = (manifest["split"] == "training") & (manifest["cv_batch"] != cv_fold)
    train_paths = manifest[train_mask]["pt_path"].tolist()

    # Validation: cv_fold
    val_mask = (manifest["split"] == "training") & (manifest["cv_batch"] == cv_fold)
    val_paths = manifest[val_mask]["pt_path"].tolist()

    # Test: all test data
    test_mask = manifest["split"] == "testing"
    test_paths = manifest[test_mask]["pt_path"].tolist()

    return (
        LazyGraphDataset(train_paths),
        LazyGraphDataset(val_paths),
        LazyGraphDataset(test_paths),
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PDB structures into graph .pt files."
    )
    parser.add_argument("--csv", default="pdb_splits_CL.csv",
                        help="Input CSV with PDB_ID, Chains_To_Keep, Binding_Residues, CV_Batch, Split")
    parser.add_argument("--out_dir", default="preprocessed",
                        help="Output directory for .pt files")
    parser.add_argument("--raw_dir", default="data/raw_pdbs",
                        help="Temporary directory for PDB downloads")
    parser.add_argument("--esm_cache", default=".esm_cache",
                        help="Cache directory for ESM2 model")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Proteins per batch for ESM processing")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Parallel download workers")
    args = parser.parse_args()

    manifest = preprocess(
        csv_path=args.csv,
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        esm_cache=args.esm_cache,
        batch_size=args.batch_size,
        device=args.device,
        max_workers=args.max_workers,
    )

    print(f"\nDone! Manifest shape: {manifest.shape}")