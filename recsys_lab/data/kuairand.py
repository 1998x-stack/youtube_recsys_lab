from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

@dataclass
class KuaiRandPaths:
    data_dir: str
    file: str  # log_random_*.csv

def load_kuairand_random(paths: KuaiRandPaths) -> pd.DataFrame:
    """
    读取 KuaiRand 随机曝光日志（log_random_*.csv），核心字段：
    user_id, video_id, date(int), time_ms(int), play_time_ms, duration_ms, is_rand(=1), ...
    我们将构造 play_ratio = play_time_ms / duration_ms，并做按用户的时间排序。
    """
    fp = f"{paths.data_dir}/{paths.file}"
    df = pd.read_csv(fp)
    # 规整字段名（不同版本列名一致性已在公开版对齐，这里仍做保守处理）
    df = df.rename(columns={
        "play_time_ms": "play_ms",
        "duration_ms": "dur_ms"
    })
    # 清理与派生
    df["play_ms"] = df["play_ms"].astype(float)
    df["dur_ms"] = df["dur_ms"].replace(0, np.nan).astype(float)  # 防止除零
    df["play_ratio"] = (df["play_ms"] / df["dur_ms"]).fillna(0.0).clip(0, 10.0)
    df = df[[
        "user_id", "video_id", "date", "time_ms", "is_rand",
        "play_ms", "dur_ms", "play_ratio"
    ]].copy()
    # 排序
    df = df.sort_values(["user_id", "date", "time_ms"])
    logger.info(f"loaded KuaiRand random log: {len(df):,} rows")
    return df

def build_user_history_stats(df: pd.DataFrame, win: int = 20) -> pd.DataFrame:
    """
    构造连续特征：用户级 rolling 观看比率统计（shift 以防泄露）
    """
    grp = df.groupby("user_id")["play_ratio"]
    mean = grp.transform(lambda x: x.shift().rolling(win, min_periods=1).mean())
    std  = grp.transform(lambda x: x.shift().rolling(win, min_periods=1).std()).fillna(0.0)
    out = df.copy()
    out["play_ratio_hist_mean"] = mean.fillna(0.0).astype(float)
    out["play_ratio_hist_std"] = std.astype(float)
    return out

def split_train_valid_by_date(df: pd.DataFrame, valid_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    简易按日期切分（最后若干天作为验证）
    """
    days = np.sort(df["date"].unique())
    n_valid = max(1, int(len(days) * valid_ratio))
    valid_days = set(days[-n_valid:])
    train = df[~df["date"].isin(valid_days)].copy()
    valid = df[df["date"].isin(valid_days)].copy()
    return train, valid
