from pathlib import Path
import torch
import torch.nn as nn
import wandb
from .metrics import compute_metrics
from src.visualize.graph_utils import (
     create_wandb_comparison_table,
     add_to_wandb_comparison_table,
)
import os
from dotenv import load_dotenv

load_dotenv()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    tag: str = "best",              # 'best' or 'latest'
) -> str:
    """
    Saves checkpoint locally then uploads to W&B as an artifact.
    Download from W&B UI → Artifacts → model-checkpoints → files tab.
    Or via API: wandb.restore('checkpoints/best.pt', run_path='...')
    """
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    path = f"{config['checkpoint_dir']}/{tag}.pt"
 
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics":     metrics,
        "config":      config,
    }, path)
 
    # Upload to W&B as a versioned artifact
    artifact = wandb.Artifact(
        name     = f"model-{wandb.run.id}",
        type     = "model",
        metadata = {"epoch": epoch, **metrics},
    )
    artifact.add_file(path)
    wandb.log_artifact(artifact, aliases=[tag, f"epoch-{epoch}"])
 
    return path


@torch.no_grad()
def build_val_table(
    model: nn.Module,
    val_samples: list[dict],
    device: torch.device,
):
    """
    Runs inference on the 5 val samples and builds the W&B comparison table.
    Uses your existing create_wandb_comparison_table / add_to_wandb_comparison_table.
    """
    pdb_paths, y_trues, y_preds = [], [], []
 
    model.eval()
    for sample in val_samples:
        graph  = sample["graph"].to(device)
        logits = model(graph)
        probs  = torch.sigmoid(logits).cpu().numpy()
 
        pdb_paths.append(sample["pdb_path"])
        y_trues.append(sample["y_true"])
        y_preds.append(probs)
 
    # Build table using your functions
    table = create_wandb_comparison_table(pdb_paths[0], y_trues[0], y_preds[0])
    for pdb_path, y_true, y_pred in zip(pdb_paths[1:], y_trues[1:], y_preds[1:]):
        add_to_wandb_comparison_table(table, pdb_path, y_true, y_pred)
 
    return table