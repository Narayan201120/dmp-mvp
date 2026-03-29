from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ToyTransformerConfig:
    vocab_size: int = 64
    max_seq_len: int = 16
    d_model: int = 32
    num_heads: int = 4
    mlp_hidden_dim: int = 64
    num_layers: int = 6

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if self.d_model < 1:
            raise ValueError("d_model must be positive")
        if self.num_heads < 1 or self.d_model % self.num_heads != 0:
            raise ValueError("num_heads must divide d_model")
        if self.mlp_hidden_dim < self.d_model:
            raise ValueError("mlp_hidden_dim must be at least d_model")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return attended


class ToyTransformerBlock(nn.Module):
    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config.d_model, config.num_heads)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(config.mlp_hidden_dim, config.d_model),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.attn_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ToyTransformer(nn.Module):
    def __init__(self, config: ToyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(ToyTransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_seq_len:
            raise ValueError("sequence length exceeds configured max_seq_len")
        positions = torch.arange(seq_len, device=input_ids.device)
        position_ids = positions.unsqueeze(0).expand_as(input_ids)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        *,
        start_layer: int = 0,
        end_layer: int | None = None,
    ) -> torch.Tensor:
        stop = len(self.blocks) if end_layer is None else end_layer
        for block in self.blocks[start_layer:stop]:
            hidden_states = block(hidden_states)
        return hidden_states

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(hidden_states))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed(input_ids)
        hidden_states = self.forward_blocks(hidden_states)
        return self.project(hidden_states)


def build_toy_transformer(
    config: ToyTransformerConfig | None = None,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> ToyTransformer:
    torch.manual_seed(seed)
    model = ToyTransformer(config or ToyTransformerConfig())
    model.to(device)
    model.eval()
    return model

