from __future__ import annotations
import argparse, os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger as L
from recsys_lab.utils.common import load_cfg, set_seed, get_device
from recsys_lab.utils.logging import setup_logger
from recsys_lab.features.quantile import fit_apply_pipeline, QuantileScaler, add_poly_feats
from recsys_lab.models.rank_tower import RankTower, weighted_bce_logits, bpr_loss

# ------- KuaiRec 点式数据集（带简易负采样近似） -------
class RankDatasetKRec(Dataset):
    def __init__(self, df: pd.DataFrame, feats: np.ndarray, neg_rate: int = 1):
        pos_X = feats.astype(np.float32)
        pos_y = np.ones((len(df), 1), dtype=np.float32)
        pos_w = df["watch_ratio"].values[:, None].astype(np.float32)

        # 简易负例：同用户历史统计下复制特征但标注0（真实应用用曝光日志）
        neg_n = len(df) * neg_rate
        neg_X = pos_X[np.random.choice(len(df), size=neg_n, replace=True)]
        neg_y = np.zeros((neg_n, 1), dtype=np.float32)
        neg_w = np.ones((neg_n, 1), dtype=np.float32)

        self.X = np.concatenate([pos_X, neg_X], axis=0)
        self.y = np.concatenate([pos_y, neg_y], axis=0)
        self.w = np.concatenate([pos_w, neg_w], axis=0)

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.float32),
                torch.tensor(self.y[i], dtype=torch.float32).squeeze(0),
                torch.tensor(self.w[i], dtype=torch.float32).squeeze(0))

# ------- KuaiRand 点式 BCE 数据集（权重=play_ms） -------
class RankDatasetKRandPoint(Dataset):
    def __init__(self, df: pd.DataFrame, feats: np.ndarray):
        self.X = feats.astype(np.float32)
        self.y = (df["play_ms"].values > 0).astype(np.float32)[:, None]
        self.w = df["play_ms"].values[:, None].astype(np.float32)  # 按观看时长加权

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X[i], dtype=torch.float32),
                torch.tensor(self.y[i], dtype=torch.float32).squeeze(0),
                torch.tensor(self.w[i], dtype=torch.float32).squeeze(0))

# ------- KuaiRand Pairwise BPR 数据集 -------
class PairDatasetKRandBPR(Dataset):
    """
    以每个正例（play_ms>0）为锚，采一个负例：同一用户当日内 is_rand=1 且 play_ms=0 的样本；若无则全局退化采样。
    """
    def __init__(self, df: pd.DataFrame, feats: np.ndarray):
        self.pos_idx = np.where(df["play_ms"].values > 0)[0]
        self.df = df.reset_index(drop=True)
        self.X = feats.astype(np.float32)

        # 建立 user-day -> 负例索引池
        self.neg_pool = {}
        key = (self.df["user_id"].values, self.df["date"].values)
        mask_neg = (self.df["is_rand"].values == 1) & (self.df["play_ms"].values <= 0)
        for i, (u, d) in enumerate(zip(*key)):
            if mask_neg[i]:
                self.neg_pool.setdefault((u, d), []).append(i)
        self.global_negs = np.where(mask_neg)[0]
        if len(self.global_negs) == 0:
            raise ValueError("No negatives found (is_rand=1 & play_ms=0)")

    def __len__(self): return len(self.pos_idx)
    def __getitem__(self, k):
        pi = self.pos_idx[k]
        u = int(self.df.at[pi, "user_id"]); d = int(self.df.at[pi, "date"])
        cand = self.neg_pool.get((u, d), None)
        if not cand:
            ni = int(np.random.choice(self.global_negs))
        else:
            ni = int(np.random.choice(cand))
        x_pos = torch.tensor(self.X[pi], dtype=torch.float32)
        x_neg = torch.tensor(self.X[ni], dtype=torch.float32)
        # 以正例观看时长作为权重（也可用比率）
        w = torch.tensor(float(self.df.at[pi, "play_ms"]), dtype=torch.float32)
        return x_pos, x_neg, w

# ----------------- 主流程 -----------------
def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    setup_logger(cfg["logging"]["log_level"])
    set_seed(cfg["seed"])
    dev = get_device(cfg["device"])
    L.info(f"device={dev}")

    src = cfg.get("data", {}).get("source", "kuairec")
    loss_type = cfg["ranking"]["loss"]

    # ==== 加载并构造连续特征 ====
    if src == "kuairec":
        # KuaiRec：与旧版相同（历史均值/方差作连续特征）
        data_dir = cfg["data"]["data_dir"]
        name = "small_matrix.csv" if cfg["data"]["use_small_matrix"] else "big_matrix.csv"
        df = pd.read_csv(os.path.join(data_dir, name))
        df = df.rename(columns={"play_duration": "play_ms", "video_duration": "dur_ms"})
        df["watch_ratio"] = (df["play_ms"] / df["dur_ms"].replace(0, np.nan)).fillna(0.0).clip(0, 10.0)
        df = df[["user_id","video_id","timestamp","date","watch_ratio"]].sort_values(["user_id","timestamp"])

        grp = df.groupby("user_id")["watch_ratio"]
        df["play_ratio_hist_mean"] = grp.transform(lambda x: x.shift().rolling(20, min_periods=1).mean()).fillna(0.0)
        df["play_ratio_hist_std"] = grp.transform(lambda x: x.shift().rolling(20, min_periods=1).std()).fillna(0.0)
        # 切分（简化：按 date 篇幅）
        days = np.sort(df["date"].unique()); valid_days = set(days[-max(1, int(len(days)*0.1)):])
        train = df[~df["date"].isin(valid_days)].copy()
        valid = df[df["date"].isin(valid_days)].copy()

        feats_train, meta = fit_apply_pipeline(train[cfg["ranking"]["cont_feats"]].values.astype(np.float64),
                                               use_poly=cfg["ranking"]["use_poly_feats"])
        scaler_q = np.array(meta["q"]); use_poly = meta["use_poly"]
        val_u = QuantileScaler(q=scaler_q).transform(valid[cfg["ranking"]["cont_feats"]].values.astype(np.float64))
        feats_valid = add_poly_feats(val_u) if use_poly else val_u[..., None]

        if loss_type == "weighted_bce":
            ds_tr = RankDatasetKRec(train, feats_train, neg_rate=1)
            ds_va = RankDatasetKRec(valid, feats_valid, neg_rate=1)
        elif loss_type == "bpr":
            # KuaiRec 无曝光负例，这里不建议做 BPR（保留为占位，使用 pointwise）
            L.warning("BPR on KuaiRec is not well-defined (no exposure negatives). Falling back to BCE.")
            ds_tr = RankDatasetKRec(train, feats_train, neg_rate=1)
            ds_va = RankDatasetKRec(valid, feats_valid, neg_rate=1)
            loss_type = "weighted_bce"

    elif src == "kuairand":
        from recsys_lab.data.kuairand import KuaiRandPaths, load_kuairand_random, build_user_history_stats, split_train_valid_by_date
        paths = KuaiRandPaths(cfg["data"]["data_dir"], cfg["data"]["file"])
        df = load_kuairand_random(paths)
        df = build_user_history_stats(df, win=20)
        train, valid = split_train_valid_by_date(df, valid_ratio=0.1)

        # 评估是否仅使用 is_rand=1（更接近去偏验证）
        if cfg["ranking"].get("use_is_rand_eval_only", False):
            valid = valid[valid["is_rand"] == 1].copy()
            L.info(f"valid restricted to is_rand=1: {len(valid):,}")

        feats_train, meta = fit_apply_pipeline(train[cfg["ranking"]["cont_feats"]].values.astype(np.float64),
                                               use_poly=cfg["ranking"]["use_poly_feats"])
        scaler_q = np.array(meta["q"]); use_poly = meta["use_poly"]
        val_u = QuantileScaler(q=scaler_q).transform(valid[cfg["ranking"]["cont_feats"]].values.astype(np.float64))
        feats_valid = add_poly_feats(val_u) if use_poly else val_u[..., None]

        if loss_type == "weighted_bce":
            ds_tr = RankDatasetKRandPoint(train, feats_train)
            ds_va = RankDatasetKRandPoint(valid, feats_valid)
        elif loss_type == "bpr":
            ds_tr = PairDatasetKRandBPR(train, feats_train)
            ds_va = PairDatasetKRandBPR(valid, feats_valid)
        else:
            raise ValueError(f"unknown ranking.loss: {loss_type}")
    else:
        raise ValueError(f"unknown data.source: {src}")

    L.info(f"train size={len(ds_tr):,}, valid size={len(ds_va):,}, loss={loss_type}")

    # ==== 模型 ====
    in_dim = (ds_tr[0][0].shape[-1] if loss_type != "bpr" else ds_tr[0][0].shape[-1])
    model = RankTower(in_dim=in_dim, tower_widths=tuple(cfg["ranking"]["tower_widths"])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ranking"]["lr"])

    # ==== 训练 & 验证 ====
    if loss_type == "weighted_bce":
        tr_loader = DataLoader(ds_tr, batch_size=cfg["ranking"]["batch_size"], shuffle=True)
        va_loader = DataLoader(ds_va, batch_size=4096, shuffle=False)

        def run_epoch(loader, train_flag=True):
            model.train() if train_flag else model.eval()
            tot = 0.0
            for X, y, w in loader:
                X, y, w = X.to(dev), y.to(dev), w.to(dev)
                with torch.set_grad_enabled(train_flag):
                    logits = model(X)
                    loss = weighted_bce_logits(logits, y, w)
                    if train_flag:
                        opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(y)
            return tot / len(loader.dataset)

        for ep in range(cfg["ranking"]["epochs"]):
            tr = run_epoch(tr_loader, True)
            va = run_epoch(va_loader, False)
            L.info(f"[rank BCE ep {ep+1}] train_wBCE={tr:.5f}  valid_wBCE={va:.5f}")

    else:  # BPR
        tr_loader = DataLoader(ds_tr, batch_size=cfg["ranking"]["batch_size"], shuffle=True)
        va_loader = DataLoader(ds_va, batch_size=4096, shuffle=False)

        def run_epoch_pair(loader, train_flag=True):
            model.train() if train_flag else model.eval()
            tot = 0.0; n = 0
            for Xp, Xn, w in loader:
                Xp, Xn, w = Xp.to(dev), Xn.to(dev), w.to(dev)
                with torch.set_grad_enabled(train_flag):
                    sp = model(Xp); sn = model(Xn)
                    loss = bpr_loss(sp, sn, w)
                    if train_flag:
                        opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(w); n += len(w)
            return tot / max(1, n)

        for ep in range(cfg["ranking"]["epochs"]):
            tr = run_epoch_pair(tr_loader, True)
            va = run_epoch_pair(va_loader, False)
            L.info(f"[rank BPR ep {ep+1}] train_BPR={tr:.5f}  valid_BPR={va:.5f}")

    # 保存
    os.makedirs(cfg["ranking"]["save_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["ranking"]["save_dir"], "rank.pt"))
    L.info(f"saved rank model to {cfg['ranking']['save_dir']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
