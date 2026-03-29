from __future__ import annotations

import torch
import torch.nn.functional as F


def split_next_token_batch(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    if input_ids.size(1) < 2:
        raise ValueError("sequence length must be at least 2 for next-token prediction")
    return input_ids[:, :-1], input_ids[:, 1:]


def next_token_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, seq]")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logit batch/sequence dimensions must match target_ids")

    vocab_size = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

