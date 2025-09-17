from __future__ import annotations
import torch, torch.nn as nn
from typing import Tuple

class CandidTower(nn.Module):
    """平均池化历史 + MLP 用户塔；物品嵌入表；点积为logit"""
    def __init__(self, n_items: int, embed_dim: int, tower_widths=(256,128)):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embed_dim)
        layers = []
        in_dim = embed_dim + 1  # + example_age (可为0)
        for h in tower_widths:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, embed_dim)]
        self.user_mlp = nn.Sequential(*layers)

    def encode_user(self, hist_ids: torch.LongTensor, example_age: torch.FloatTensor) -> torch.Tensor:
        # hist_ids: [B, H]
        h = self.item_emb(hist_ids).mean(dim=1)  # [B, D]
        x = torch.cat([h, example_age], dim=1)
        u = self.user_mlp(x)                     # [B, D]
        return nn.functional.normalize(u, dim=1)

    def encode_item(self) -> torch.Tensor:
        W = self.item_emb.weight                 # [N, D]
        return nn.functional.normalize(W, dim=1)

    def dot(self, u: torch.Tensor, item_ids: torch.LongTensor) -> torch.Tensor:
        v = self.item_emb(item_ids)              # [B,K,D] if 2D ids
        v = nn.functional.normalize(v, dim=-1)
        return (u.unsqueeze(1) * v).sum(-1)      # [B,K]
