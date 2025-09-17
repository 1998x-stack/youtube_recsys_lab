from __future__ import annotations
import numpy as np

class UnigramCandidateSampler:
    """基于物品频次的一元分布采样（近似重要性分布）；返回 [B, K] 负例"""
    def __init__(self, item_freq: np.ndarray, distortion: float = 0.75, rng: np.random.RandomState | None = None):
        p = item_freq.astype(np.float64)
        p = p ** distortion
        p = p / p.sum()
        self.p = p
        self.n_items = len(p)
        self.rng = rng or np.random.RandomState(42)

    def sample(self, batch_size: int, k: int) -> np.ndarray:
        return self.rng.choice(self.n_items, size=(batch_size, k), replace=True, p=self.p)

    def prob(self, item_ids: np.ndarray) -> np.ndarray:
        return self.p[item_ids]
