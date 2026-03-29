from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from daemon.messaging import MessageBus
from daemon.state import NodeState
from training.protocol import WindowSession, materialize_tensor_boundary_state, payload_as_tensor
from training.shard import TransformerShard


@dataclass(slots=True)
class TrainingLoop:
    bus: MessageBus
    state: NodeState
    shard: TransformerShard | None = None

    async def train_batch(self, batch: Any | None = None) -> dict[str, Any]:
        metric = {"node_id": self.state.node_id, "shard_id": self.state.shard_id, "version": self.state.version}
        await self.bus.publish("training.metrics", metric)
        return metric

    async def emit_gradient_update(self, payload: dict[str, Any]) -> None:
        await self.bus.publish("training.updates", payload)

    async def run_window_forward(
        self,
        session: WindowSession,
        *,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.shard is None:
            raise RuntimeError("training loop cannot execute without an attached shard")

        if input_ids is not None:
            output = self.shard(input_ids=input_ids)
        else:
            output = self.shard(hidden_states=hidden_states)

        await self.bus.publish(
            "training.metrics",
            {
                "node_id": self.state.node_id,
                "shard_id": self.state.shard_id,
                "version": session.spec.version,
                "kind": "window_forward",
            },
        )

        if self.shard.is_last_shard:
            return output

        target_shard = self.state.shard_id + 1
        payload = materialize_tensor_boundary_state(
            session,
            source_shard=self.state.shard_id,
            target_shard=target_shard,
            tensor=output,
        )
        session.exchange_boundary_state(payload)
        await self.bus.publish(
            "training.boundaries",
            {
                "version": payload.version,
                "source_shard": payload.source_shard,
                "target_shard": payload.target_shard,
                "shape": tuple(payload.tensor.shape),
            },
        )
        return payload_as_tensor(payload, device=output.device, dtype=output.dtype)
