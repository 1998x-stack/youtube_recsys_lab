from __future__ import annotations
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader
from loguru import logger as L
from recsys_lab.utils.common import load_cfg, set_seed, get_device
from recsys_lab.utils.logging import setup_logger
from recsys_lab.data.kuairec import KuairecPaths, load_interactions, CandidDataset, build_item_freq
from recsys_lab.data.samplers import UnigramCandidateSampler
from recsys_lab.models.candid_tower import CandidTower
from recsys_lab.losses.sampled_softmax import sampled_softmax_loss
from recsys_lab.metrics.candidate import recall_at_k, map_at_k
from tqdm import tqdm

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    setup_logger(cfg["logging"]["log_level"]); set_seed(cfg["seed"])
    dev = get_device(cfg["device"]); L.info(f"device={dev}")

    # 1) load data
    paths = KuairecPaths(cfg["data"]["data_dir"], cfg["data"]["use_small_matrix"])
    df = load_interactions(paths, cfg["data"]["min_watch_ratio"])
    n_items = int(df["video_id"].max()) + 1

    ds = CandidDataset(
        df=df,
        history_len=cfg["data"]["history_len"],
        t_min=cfg["data"]["train_days"][0],
        t_max=cfg["data"]["train_days"][1],
        use_example_age=cfg["data"]["use_example_age"],
        n_items=n_items
    )
    loader = DataLoader(ds, batch_size=cfg["candidate"]["batch_size"], shuffle=True, drop_last=True, num_workers=0)

    # 2) sampler + model
    item_freq = build_item_freq(df, n_items)
    sampler = UnigramCandidateSampler(item_freq)
    model = CandidTower(n_items=n_items, embed_dim=cfg["candidate"]["embed_dim"],
                        tower_widths=tuple(cfg["candidate"]["tower_widths"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["candidate"]["lr"], weight_decay=cfg["candidate"]["weight_decay"])

    # 3) train
    model.train()
    for ep in range(cfg["candidate"]["epochs"]):
        pbar = tqdm(loader, desc=f"candid/ep{ep+1}")
        for hist, label, ex_age in pbar:
            hist, label, ex_age = hist.to(dev), label.to(dev), ex_age.to(dev)
            B = hist.size(0)
            neg_ids = torch.from_numpy(sampler.sample(B, cfg["candidate"]["neg_k"])).to(dev)
            q_pos = torch.from_numpy(sampler.prob(label.cpu().numpy())).unsqueeze(1).to(dev)
            q_neg = torch.from_numpy(sampler.prob(neg_ids.cpu().numpy())).to(dev)

            u = model.encode_user(hist, ex_age)                        # [B,D]
            pos_logits = model.dot(u, label.unsqueeze(1))              # [B,1]
            neg_logits = model.dot(u, neg_ids)                         # [B,K]
            loss = sampled_softmax_loss(pos_logits, neg_logits, q_pos, q_neg)

            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))

        # quick val on a small slice
        model.eval()
        with torch.no_grad():
            # 构造一个小验证切片
            ds_val = CandidDataset(df, cfg["data"]["history_len"], cfg["data"]["eval_days"][0],
                                   cfg["data"]["eval_days"][1], cfg["data"]["use_example_age"], n_items)
            val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)
            users, gts = [], []
            for hist, label, ex_age in val_loader:
                u = model.encode_user(hist.to(dev), ex_age.to(dev)).cpu().numpy()
                users.append(u); gts.append(label.numpy())
            U = np.concatenate(users); GT = np.concatenate(gts)
            V = model.encode_item().detach().cpu().numpy()
            r100 = recall_at_k(U, V, GT, k=100); map100 = map_at_k(U, V, GT, k=100)
            L.info(f"[ep {ep+1}] recall@100={r100:.4f} map@100={map100:.4f}")
        model.train()

    # 4) save
    os.makedirs(cfg["candidate"]["save_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["candidate"]["save_dir"], "candid.pt"))
    torch.save(model.item_emb.weight.detach().cpu(), os.path.join(cfg["candidate"]["save_dir"], "item_emb.pt"))
    L.info("saved model & item vectors")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
