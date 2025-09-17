from __future__ import annotations
import numpy as np
import faiss

def recall_at_k(user_vecs: np.ndarray, item_vecs: np.ndarray, gt_ids: np.ndarray, k: int = 100) -> float:
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    faiss.normalize_L2(item_vecs)
    index.add(item_vecs.astype(np.float32))
    faiss.normalize_L2(user_vecs)
    _, I = index.search(user_vecs.astype(np.float32), k)  # [B, k]
    hits = np.array([1 if gt in I[i] else 0 for i, gt in enumerate(gt_ids)])
    return hits.mean()

def map_at_k(user_vecs: np.ndarray, item_vecs: np.ndarray, gt_ids: np.ndarray, k: int = 100) -> float:
    index = faiss.IndexFlatIP(item_vecs.shape[1])
    faiss.normalize_L2(item_vecs); index.add(item_vecs.astype(np.float32))
    faiss.normalize_L2(user_vecs)
    _, I = index.search(user_vecs.astype(np.float32), k)
    ap = []
    for i, gt in enumerate(gt_ids):
        rank = np.where(I[i] == gt)[0]
        ap.append(1.0 / (rank[0] + 1) if len(rank) else 0.0)
    return float(np.mean(ap))
