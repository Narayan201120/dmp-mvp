from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn

from training.model_factory import ToyTransformer


@dataclass(frozen=True, slots=True)
class ShardSpec:
    shard_id: int
    start_layer: int
    end_layer: int


def contiguous_shards(num_layers: int, num_shards: int) -> list[ShardSpec]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if num_shards > num_layers:
        raise ValueError("num_shards cannot exceed num_layers")

    base_size, remainder = divmod(num_layers, num_shards)
    start = 0
    shards: list[ShardSpec] = []

    for shard_id in range(num_shards):
        width = base_size + (1 if shard_id < remainder else 0)
        end = start + width
        shards.append(ShardSpec(shard_id=shard_id, start_layer=start, end_layer=end))
        start = end

    return shards


@dataclass(frozen=True, slots=True)
class ShardedForwardResult:
    logits: torch.Tensor
    boundary_states: dict[tuple[int, int], torch.Tensor]


class TransformerShard(nn.Module):
    def __init__(
        self,
        *,
        spec: ShardSpec,
        blocks: list[nn.Module],
        token_embedding: nn.Embedding | None = None,
        position_embedding: nn.Embedding | None = None,
        final_norm: nn.LayerNorm | None = None,
        lm_head: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.blocks = nn.ModuleList(blocks)
        self.token_embedding = token_embedding
        self.position_embedding = position_embedding
        self.final_norm = final_norm
        self.lm_head = lm_head

    @property
    def is_first_shard(self) -> bool:
        return self.token_embedding is not None and self.position_embedding is not None

    @property
    def is_last_shard(self) -> bool:
        return self.final_norm is not None and self.lm_head is not None

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.is_first_shard:
            if input_ids is None:
                raise ValueError("first shard requires input_ids")
            hidden_states = _embed_inputs(input_ids, self.token_embedding, self.position_embedding)
        elif hidden_states is None:
            raise ValueError("non-initial shards require hidden_states")

        assert hidden_states is not None
        for block in self.blocks:
            hidden_states = block(hidden_states)

        if self.is_last_shard:
            assert self.final_norm is not None
            assert self.lm_head is not None
            return self.lm_head(self.final_norm(hidden_states))
        return hidden_states


def build_transformer_shards(model: ToyTransformer, num_shards: int) -> tuple[list[TransformerShard], list[ShardSpec]]:
    shard_specs = contiguous_shards(len(model.blocks), num_shards)
    last_index = len(shard_specs) - 1
    shards: list[TransformerShard] = []

    for index, spec in enumerate(shard_specs):
        blocks = [copy.deepcopy(block) for block in model.blocks[spec.start_layer : spec.end_layer]]
        shards.append(
            TransformerShard(
                spec=spec,
                blocks=blocks,
                token_embedding=copy.deepcopy(model.token_embedding) if index == 0 else None,
                position_embedding=copy.deepcopy(model.position_embedding) if index == 0 else None,
                final_norm=copy.deepcopy(model.final_norm) if index == last_index else None,
                lm_head=copy.deepcopy(model.lm_head) if index == last_index else None,
            )
        )

    return shards, shard_specs


def forward_sharded(shards: list[TransformerShard], input_ids: torch.Tensor) -> ShardedForwardResult:
    if not shards:
        raise ValueError("at least one shard is required")

    hidden_states: torch.Tensor | None = None
    boundary_states: dict[tuple[int, int], torch.Tensor] = {}
    logits: torch.Tensor | None = None

    for index, shard in enumerate(shards):
        if index == 0:
            output = shard(input_ids=input_ids)
        else:
            output = shard(hidden_states=hidden_states)

        if index == len(shards) - 1:
            logits = output
        else:
            hidden_states = output
            boundary_states[(shard.spec.shard_id, shards[index + 1].spec.shard_id)] = output.detach().clone()

    assert logits is not None
    return ShardedForwardResult(logits=logits, boundary_states=boundary_states)


def _embed_inputs(
    input_ids: torch.Tensor,
    token_embedding: nn.Embedding | None,
    position_embedding: nn.Embedding | None,
) -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, seq]")
    assert token_embedding is not None
    assert position_embedding is not None
    seq_len = input_ids.size(1)
    if seq_len > position_embedding.num_embeddings:
        raise ValueError("sequence length exceeds the shard position embedding capacity")
    positions = torch.arange(seq_len, device=input_ids.device)
    position_ids = positions.unsqueeze(0).expand_as(input_ids)
    return token_embedding(input_ids) + position_embedding(position_ids)

