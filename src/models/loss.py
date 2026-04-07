import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    """
    Focal Loss: FL(p) = -α(1-p)^γ log(p)
 
    For binding site prediction:
    - Binding residues are ~5-15% of all residues → heavy imbalance
    - γ=2.0 standard starting point; increase to 3-4 if model ignores binding sites
    - α=0.25 standard; increase toward 0.5 if recall on binding sites is too low
 
    Fine-tuning guidance:
    - Start: α=0.25, γ=2.0
    - If recall too low (missing binding sites): increase α toward 0.5
    - If precision too low (too many false positives): decrease α toward 0.1
    - If model collapses to predicting all zeros: increase γ
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE gives log(p) and log(1-p) stably
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
 
        # p_t: probability of the *correct* class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
 
        # α_t: weight for the correct class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
 
        # Focal weight: (1-p_t)^γ — down-weights easy examples
        focal_weight = (1 - p_t) ** self.gamma
 
        loss = alpha_t * focal_weight * bce
 
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss