from __future__ import annotations
import numpy as np

def weighted_pairwise_loss(scores_pos: np.ndarray, scores_neg: np.ndarray, wt_pos: np.ndarray) -> float:
    """Watch-time weighted pairwise logistic loss, stable for extreme score gaps."""
    margins = np.asarray(scores_pos) - np.asarray(scores_neg)
    losses = np.logaddexp(0.0, -margins) * np.asarray(wt_pos)
    return float(np.mean(losses))
