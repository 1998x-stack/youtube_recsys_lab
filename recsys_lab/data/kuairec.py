from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger
import os

@dataclass
class KuairecPaths:
    data_dir: str
    use_small: bool

    def interactions_csv(self) -> str:
        name = "small_matrix.csv" if self.use_small else "big_matrix.csv"
        return os.path.join(self.data_dir, name)

def _reindex_ids(df: pd.DataFrame) -> pd.DataFrame:
    """将 user_id / video_id 连续化，从 0 开始"""
    u_codes = pd.Categorical(df["user_id"])
    i_codes = pd.Categorical(df["video_id"])
    df["user_id"] = u_codes.codes.astype(np.int64)
    df["video_id"] = i_codes.codes.astype(np.int64)
    logger.info(f"reindexed users={u_codes.categories.size:,}, items={i_codes.categories.size:,}")
    return df

def load_interactions(paths: KuairecPaths, min_watch_ratio: float) -> pd.DataFrame:
    df = pd.read_csv(paths.interactions_csv())
    # 字段名适配
    df = df.rename(columns={"play_duration": "play_ms", "video_duration": "dur_ms"})
    # 关键字段存在性检查
    for col in ["user_id", "video_id", "timestamp", "date", "play_ms", "dur_ms"]:
        if col not in df.columns:
            raise ValueError(f"missing column in KuaiRec: {col}")
    # 派生
    df["watch_ratio"] = (df["play_ms"] / df["dur_ms"].replace(0, np.nan)).fillna(0.0).clip(0, 10.0)
    df = df[df["watch_ratio"] >= min_watch_ratio].copy()
    df = df[["user_id", "video_id", "timestamp", "date", "watch_ratio"]]
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df = _reindex_ids(df)
    logger.info(f"loaded KuaiRec interactions: {len(df):,}")
    return df

def build_item_freq(df: pd.DataFrame, n_items: int) -> np.ndarray:
    cnt = np.bincount(df["video_id"].values, minlength=n_items)
    return cnt + 1  # add-one 平滑

class CandidDataset(Dataset):
    """候选塔训练样本：按用户时间序列构造 (hist -> next)"""
    def __init__(self, df: pd.DataFrame, history_len: int, t_min: int, t_max: int,
                 use_example_age: bool, n_items: int):
        self.samples: List[Dict] = []
        max_ts = int(df["timestamp"].max())
        gb = df.groupby("user_id", sort=False)
        for uid, g in gb:
            g = g[(g["date"] >= t_min) & (g["date"] <= t_max)]
            vids = g["video_id"].values
            ts = g["timestamp"].values
            if len(vids) <= history_len:
                continue
            for i in range(history_len, len(vids)):
                hist = vids[i-history_len:i]
                label = vids[i]
                ex_age = float(max_ts - ts[i]) if use_example_age else 0.0
                self.samples.append({
                    "user_id": int(uid),
                    "hist": hist.astype(np.int64),
                    "label": int(label),
                    "example_age": ex_age
                })
        self.n_items = n_items

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (torch.tensor(s["hist"], dtype=torch.long),
                torch.tensor(s["label"], dtype=torch.long),
                torch.tensor([s["example_age"]], dtype=torch.float32))
