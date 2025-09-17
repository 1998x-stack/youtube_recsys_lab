from __future__ import annotations
import torch, torch.nn as nn

class RankTower(nn.Module):
    """排序塔：连续特征(已做分位数/多项式) + 可扩展离散ID嵌入"""
    def __init__(self, in_dim: int, tower_widths=(256,128)):
        super().__init__()
        layers = []
        d = in_dim
        for h in tower_widths:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # raw score

def weighted_bce_logits(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    标准 BPR：-log σ(s_pos - s_neg)；可选用正例权重（如观看时长）加权
    """
    diff = pos_scores - neg_scores
    loss = -torch.log(torch.sigmoid(diff) + 1e-12)
    if weights is not None:
        loss = loss * weights
    return loss.mean()
