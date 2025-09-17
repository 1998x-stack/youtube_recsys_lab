from __future__ import annotations
import argparse, os, numpy as np, faiss, torch, time
from loguru import logger as L
from recsys_lab.utils.common import load_cfg

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    item_vecs = torch.load(cfg["ann"]["item_vecs"])  # [N,D] tensor
    V = item_vecs.detach().cpu().numpy().astype(np.float32)
    faiss.normalize_L2(V)
    idx = faiss.IndexFlatIP(V.shape[1])
    t0 = time.time(); idx.add(V); t1 = time.time()
    L.info(f"FAISS add {len(V)} vecs, time={t1-t0:.3f}s")
    faiss.write_index(idx, cfg["ann"]["faiss_index"])
    L.info(f"wrote {cfg['ann']['faiss_index']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
