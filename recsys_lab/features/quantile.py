from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

class QuantileScaler:
    """简易分位数归一化器：把连续特征映射到 [0,1)，并可生成 x^2 与 sqrt(x)"""
    def __init__(self, q: np.ndarray | None = None):
        self.q = q  # shape [101] for 0..100 percentiles

    def fit(self, x: np.ndarray):
        x = x[~np.isnan(x)]
        self.q = np.percentile(x, np.arange(101))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.q is not None, "call fit() first"
        # piecewise linear CDF approximation
        ranks = np.searchsorted(self.q, x, side="left")
        ranks = np.clip(ranks, 1, len(self.q)-1)
        left = self.q[ranks-1]; right = self.q[ranks]
        frac = np.where(right > left, (x-left)/(right-left), 0.0)
        u = (ranks-1 + frac) / 100.0
        return np.clip(u, 0.0, 1.0)

def add_poly_feats(u: np.ndarray) -> np.ndarray:
    """拼接 x, x^2, sqrt(x)"""
    return np.stack([u, u**2, np.sqrt(np.clip(u, 0, 1e9))], axis=-1)

def fit_apply_pipeline(x: np.ndarray, use_poly: bool) -> Tuple[np.ndarray, Dict]:
    scaler = QuantileScaler().fit(x)
    u = scaler.transform(x)
    feats = add_poly_feats(u) if use_poly else u[..., None]
    meta = {"q": scaler.q.tolist(), "use_poly": use_poly}
    return feats, meta
