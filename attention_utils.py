import math
from typing import Tuple

import torch
from torch.nn import functional as F


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    scale=None,
    return_attn=False,
    attn_top_k=1.0,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Uses naive PyTorch sdpa implementation if we need to return_attn. Otherwise use the optimized version.

    The naive implementation will be optimized later.
    """
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    top_k = (
        S if L > 1 else int(attn_top_k * S)
    )  # We use full attention during prefill (L > 1)
    if not return_attn and top_k == S:
        return (
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=scale,
            ),
            None,
        )
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if attn_mask is not None:
        assert top_k == S, "Top-k attention not supported with masks."
        attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        attn_weight += attn_bias

    if top_k < S:
        _, top_k_idxs = attn_weight.topk(top_k, dim=-1)
        value = value.gather(
            -2, top_k_idxs.view(B, H, top_k, 1).expand(-1, -1, -1, value.shape[-1])
        )
        attn_weight = attn_weight.gather(-1, top_k_idxs)

    attn_prob = torch.softmax(attn_weight, dim=-1)
    attn_prob = torch.dropout(attn_prob, dropout_p, train=True)
    return attn_prob @ value, attn_prob


def scaled_dot_product_attention_biased(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    scale=None,
    return_attn=False,
    attn_top_k=1.0,
    importance_scores=None,  # NEW: [B, H, S] scores for key tokens
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """
    Compute scaled dot-product attention, optionally with importance score bias on attention weights.

    Args:
        query, key, value: tensors of shape [B, H, L, D] and [B, H, S, D]
        attn_mask: optional mask (e.g., causal)
        dropout_p: dropout probability
        scale: optional attention scaling factor
        return_attn: if True, return attention weights
        attn_top_k: optional top-k attention pruning (0 < attn_top_k <= 1)
        importance_scores: optional importance scores [B, H, S] to bias attention logits

    Returns:
        attention_output: [B, H, L, D]
        attention_weights (if return_attn=True): [B, H, L, S]
    """
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    top_k = S if L > 1 else int(attn_top_k * S)  # Full attention for prefill

    if not return_attn and top_k == S:
        return (
            F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=scale,
            ),
            None,
        )

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor  # [B, H, L, S]

    # Add importance score bias
    if importance_scores is not None:
        attn_weight += torch.log(importance_scores.clamp(min=1e-8))

    if attn_mask is not None:
        assert top_k == S, "Top-k attention not supported with masks."
        attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        attn_weight += attn_bias

    if top_k < S:
        _, top_k_idxs = attn_weight.topk(top_k, dim=-1)
        value = value.gather(
            -2, top_k_idxs.view(B, H, top_k, 1).expand(-1, -1, -1, value.shape[-1])
        )
        attn_weight = attn_weight.gather(-1, top_k_idxs)

    attn_prob = torch.softmax(attn_weight, dim=-1)
    # remove dropout for training
    # attn_prob = torch.dropout(attn_prob, dropout_p, train=True)

    return attn_prob @ value, attn_prob if return_attn else None
