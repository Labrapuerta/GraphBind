import torch
import numpy as np
import wandb
import os
from dotenv import load_dotenv
from typing import Optional

# Handle both relative and absolute imports
try:
    from .loss import FocalLoss
    from .metrics import compute_metrics
    from ..visualize.graph_utils import (
        create_wandb_comparison_table,
        add_to_wandb_comparison_table,
    )
except ImportError:
    from src.models.loss import FocalLoss
    from src.models.metrics import compute_metrics
    from src.visualize.graph_utils import (
        create_wandb_comparison_table,
        add_to_wandb_comparison_table,
    )

load_dotenv()


def train(
    model,
    train_loader,
    val_loader,
    config,
    test_loader=None,
    test_pdb_paths: Optional[list[str]] = None,
    device: Optional[str] = None,
):
    """
    Training loop with wandb logging.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dict with keys: lr, epochs, focal_alpha, focal_gamma, log_table_every
        test_loader: Optional DataLoader for test data (for visualization)
        test_pdb_paths: Optional list of PDB paths corresponding to test_loader samples
        device: Device to use (defaults to cuda if available)

    Returns:
        Trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])

    log_table_every = config.get("log_table_every", 5)
    best_val_f1 = 0.0

    for epoch in range(1, config["epochs"] + 1):

        # ── 1. TRAIN ──────────────────────────────────────────────────────
        model.train()
        train_logits, train_targets, train_losses = [], [], []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_logits.append(logits.detach().cpu())
            train_targets.append(batch.y.detach().cpu())
            train_losses.append(loss.item())

        scheduler.step()

        # ── 2. VALIDATE ───────────────────────────────────────────────────
        model.eval()
        val_logits, val_targets, val_losses = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)

                val_logits.append(logits.cpu())
                val_targets.append(batch.y.cpu())
                val_losses.append(loss.item())

        # ── 3. COMPUTE METRICS ────────────────────────────────────────────
        train_logits = torch.cat(train_logits)
        train_targets = torch.cat(train_targets)
        val_logits = torch.cat(val_logits)
        val_targets = torch.cat(val_targets)

        train_metrics = compute_metrics(train_logits, train_targets)
        val_metrics = compute_metrics(val_logits, val_targets)

        # ── 4. LOG METRICS ────────────────────────────────────────────────
        log_dict = {
            # Losses
            "train/loss": np.mean(train_losses),
            "val/loss": np.mean(val_losses),
            # Threshold-independent (ranking quality)
            "train/auroc": train_metrics["auroc"],
            "val/auroc": val_metrics["auroc"],
            "train/auprc": train_metrics["auprc"],
            "val/auprc": val_metrics["auprc"],
            # Threshold-dependent (classification quality)
            "train/precision": train_metrics["precision"],
            "val/precision": val_metrics["precision"],
            "train/recall": train_metrics["recall"],
            "val/recall": val_metrics["recall"],
            "train/f1": train_metrics["f1"],
            "val/f1": val_metrics["f1"],
            "train/mcc": train_metrics["mcc"],
            "val/mcc": val_metrics["mcc"],
            "train/specificity": train_metrics["specificity"],
            "val/specificity": val_metrics["specificity"],
            # Confusion matrix breakdown
            "val/tp": val_metrics["tp"],
            "val/fp": val_metrics["fp"],
            "val/tn": val_metrics["tn"],
            "val/fn": val_metrics["fn"],
            # Class balance — sanity check
            "val/binding_frac": val_metrics["binding_frac"],
            "val/pred_frac": val_metrics["pred_frac"],
            # Optimizer state
            "lr": scheduler.get_last_lr()[0],
        }

        # ── 5. LOG VISUALIZATION TABLE EVERY N EPOCHS ─────────────────────
        if (
            epoch % log_table_every == 0
            and test_loader is not None
            and test_pdb_paths is not None
        ):
            table = _create_test_visualization_table(
                model, test_loader, test_pdb_paths, epoch, device
            )
            if table is not None:
                log_dict["test/predictions"] = table

        wandb.log(log_dict, step=epoch)

        # ── 6. SAVE BEST MODEL ────────────────────────────────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "best_model.pt")

        # Print progress
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {np.mean(train_losses):.4f} | "
            f"Val Loss: {np.mean(val_losses):.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f}"
        )

    return model


def _create_test_visualization_table(
    model, test_loader, test_pdb_paths, epoch, device, max_samples=5
):
    """
    Create a wandb comparison table for test set predictions.

    Args:
        model: Trained model
        test_loader: DataLoader for test data
        test_pdb_paths: List of PDB paths corresponding to test samples
        epoch: Current epoch number
        device: Device for inference
        max_samples: Maximum number of samples to include in table

    Returns:
        wandb.Table or None if no valid samples
    """
    model.eval()
    table = None
    sample_count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if sample_count >= max_samples:
                break

            # Handle batched vs single samples
            if hasattr(batch, "ptr"):
                # Batched data - process first sample only
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)

                # Get first sample in batch
                start_idx = 0
                end_idx = batch.ptr[1].item() if len(batch.ptr) > 1 else len(batch.y)

                y_true = batch.y[start_idx:end_idx].cpu()
                y_pred = probs[start_idx:end_idx].cpu()
            else:
                # Single sample
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits)

                y_true = batch.y.cpu()
                y_pred = probs.cpu()

            # Get corresponding PDB path
            if i < len(test_pdb_paths):
                pdb_path = test_pdb_paths[i]

                try:
                    if table is None:
                        table = create_wandb_comparison_table(
                            pdb_path, epoch, y_true, y_pred, include_molecules=True
                        )
                    else:
                        add_to_wandb_comparison_table(
                            table, pdb_path, epoch, y_true, y_pred, include_molecules=True
                        )
                    sample_count += 1
                except Exception as e:
                    print(f"Warning: Could not create visualization for {pdb_path}: {e}")
                    continue

    return table
