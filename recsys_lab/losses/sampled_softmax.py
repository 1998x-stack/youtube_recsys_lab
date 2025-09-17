from __future__ import annotations
import torch
import torch.nn.functional as F

def sampled_softmax_loss(user_logits_pos: torch.Tensor,
                         user_logits_neg: torch.Tensor,
                         q_pos: torch.Tensor,
                         q_neg: torch.Tensor) -> torch.Tensor:
    """带重要性校正的 sampled softmax
    Args:
        user_logits_pos: [B, 1]  用户与正样本的点积
        user_logits_neg: [B, K] 用户与K个负样本的点积
        q_pos:  正样本在采样分布 Q 下的概率 [B,1]
        q_neg:  负样本在 Q 下的概率 [B,K]
    """
    # importance correction: log p - log q
    pos = user_logits_pos - torch.log(q_pos + 1e-12)
    neg = user_logits_neg - torch.log(q_neg + 1e-12)
    # log-sum-exp over pos+negs
    lse = torch.logsumexp(torch.cat([pos, neg], dim=1), dim=1, keepdim=True)  # [B,1]
    # -log softmax on positive
    nll = (lse - pos).mean()
    return nll
