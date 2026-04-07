from torch_geometric.data import Data
from pathlib import Path
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler


dataset_dir = Path("data/dataset")
pdb_dir     = Path("data/pdb_test")
df = pd.read_csv("data/training_split.csv")


class ProteinGraphDataset(Dataset):
    """
    Loads .pt graph files named {pdb_id}_{chain}.pt from DATASET_DIR.
    df must have columns: pdb_id, chain, split, n_residues.
    """
 
    def __init__(self, df: pd.DataFrame, dataset_dir: Path, pdb_dir: Path):
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.pdb_dir = pdb_dir
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx) -> Data:
        row   = self.df.iloc[idx]
        path  = self.dataset_dir / f"{row['pdb_id']}_{row['chain']}.pt"

        # PyG Data objects contain custom classes not in PyTorch's default
        # safe globals list — weights_only=False is safe here since these
        # are your own generated files
        graph = torch.load(path, map_location="cpu", weights_only=False)

        graph.pdb_id = f"{row['pdb_id']}_{row['chain']}"
        return graph
 
    def get_pdb_path(self, idx: int) -> Path:
        """PDB file path for visualization — only exists for val and test."""
        row = self.df.iloc[idx]
        return self.pdb_dir / f"{row['pdb_id']}_{row['chain']}.pdb"


class SortedSampler(torch.utils.data.Sampler):
    """
    Yields individual indices in size-sorted batch order.
    PyG's DataLoader then groups them into batches using its own
    collate function — which correctly handles Data objects.

    Different from batch_sampler: we yield single indices, not lists.
    DataLoader collects batch_size consecutive indices into a batch.
    Since our indices are already sorted into size-similar groups,
    each batch naturally contains similarly-sized proteins.
    """
    def __init__(
        self,
        sizes: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.sizes      = np.array(sizes)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.rng        = np.random.default_rng(seed)

    def __iter__(self):
        sorted_idx = np.argsort(self.sizes)
        batches    = [
            sorted_idx[i : i + self.batch_size].tolist()
            for i in range(0, len(sorted_idx), self.batch_size)
        ]
        if self.shuffle:
            self.rng.shuffle(batches)

        # Yield individual indices — DataLoader collects batch_size of them
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.sizes)
    
def _make_loader(
    df: pd.DataFrame,
    dataset_dir: Path,
    pdb_dir: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[PyGDataLoader, ProteinGraphDataset]:
    ds = ProteinGraphDataset(df, dataset_dir, pdb_dir)

    # Use PyG's DataLoader directly — it knows how to collate Data objects
    # No custom sampler needed, shuffle handles randomness
    loader = PyGDataLoader(
        ds,
        batch_size   = batch_size,
        shuffle      = shuffle,
        num_workers  = num_workers,
        pin_memory   = pin_memory,
    )
    return loader, ds
 
 
def make_loaders(
    df: pd.DataFrame,
    dataset_dir: Path,
    pdb_dir: Path,  
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict:
    """
    Creates train, val, and test loaders from the split CSV.
 
    Returns a dict with:
        train_loader  — shuffled, sorted batches
        val_loader    — sorted, no shuffle
        test_loader   — sorted, no shuffle
        val_dataset   — ProteinGraphDataset (use for W&B samples)
        test_dataset  — ProteinGraphDataset
        stats         — sizes for logging
    """
    train_df = df[df["split"] == "training"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "testing"].reset_index(drop=True)
 
    train_loader, _        = _make_loader(train_df, dataset_dir, pdb_dir, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    val_loader,   val_ds   = _make_loader(val_df,   dataset_dir, pdb_dir, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader,  test_ds  = _make_loader(test_df,  dataset_dir, pdb_dir, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
 
    stats = {
        "n_train":       len(train_df),
        "n_val":         len(val_df),
        "n_test":        len(test_df),
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
        "test_batches":  len(test_loader),
    }
    _print_summary(stats)
 
    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "val_dataset":  val_ds,
        "test_dataset": test_ds,
        "stats":        stats,
    }
 
 
def _print_summary(stats: dict):
    print(f"\n{'='*50}")
    print(f"  train : {stats['n_train']:>5} proteins | {stats['train_batches']:>4} batches")
    print(f"  val   : {stats['n_val']:>5} proteins | {stats['val_batches']:>4} batches")
    print(f"  test  : {stats['n_test']:>5} proteins | {stats['test_batches']:>4} batches")
    print(f"{'='*50}\n")


def get_val_samples(
    val_dataset: ProteinGraphDataset,
    n: int = 5,
    seed: int = None,
) -> list[dict]:
    """
    Returns n random val samples that have a matching .pdb file in PDB_DIR.
    Filters missing files before sampling so your W&B table never errors.
 
    Each returned dict:
        pdb_id   : str         — e.g. '966c_A'
        pdb_path : str         — absolute path to .pdb for W&B 3D viewer
        graph    : Data        — full PyG graph (x, edge_index, pos, y)
        y_true   : np.ndarray  — (N,) ground truth binding labels
    """
    # Only sample from rows that actually have a PDB file on disk
    valid_idx = [
        i for i in range(len(val_dataset))
        if val_dataset.get_pdb_path(i).exists()
    ]
 
    if not valid_idx:
        print("[warning] No PDB files found in val set — skipping W&B visualization")
        return []
 
    rng    = np.random.default_rng(seed)
    chosen = rng.choice(valid_idx, size=min(n, len(valid_idx)), replace=False)
 
    samples = []
    for idx in chosen:
        graph = val_dataset[idx]
        samples.append({
            "pdb_id":   graph.pdb_id,
            "pdb_path": str(val_dataset.get_pdb_path(idx)),
            "graph":    graph,
            "y_true":   graph.y.numpy(),
        })
    return samples