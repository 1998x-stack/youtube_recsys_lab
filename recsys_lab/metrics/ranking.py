from __future__ import annotations
import numpy as np

def weighted_pairwise_loss(scores_pos: np.ndarray, scores_neg: np.ndarray, wt_pos: np.ndarray) -> float:
    """简易 pairwise：-log(sigmoid(s_pos - s_neg))，正例按观看时长加权"""
    x = -np.log(1.0 / (1.0 + np.exp(-(scores_pos - scores_neg))) + 1e-12) * wt_pos
    return float(np.mean(x))
