from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    probs  = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)
 
    has_both = len(np.unique(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
 
    return {
        "auroc":       roc_auc_score(y_true, probs) if has_both else 0.5,
        "auprc":       average_precision_score(y_true, probs) if has_both else 0.0,
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "mcc":         matthews_corrcoef(y_true, y_pred) if has_both else 0.0,
        "specificity": tn / (tn + fp + 1e-8),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "binding_frac": float(y_true.mean()),
        "pred_frac":    float(y_pred.mean()),
    }



