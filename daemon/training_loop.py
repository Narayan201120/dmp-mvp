from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from daemon.messaging import MessageBus
from daemon.state import NodeState
from training.protocol import BoundaryPayload, WindowSession, materialize_tensor_boundary_state
from training.shard import TransformerShard


@dataclass(slots=True)
class WindowForwardResult:
    tensor: torch.Tensor
    boundary_payload: BoundaryPayload | None = None


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
        retain_graph_tensor: bool = False,
    ) -> WindowForwardResult:
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
            return WindowForwardResult(tensor=output)

        target_shard = self.state.shard_id + 1
        payload = materialize_tensor_boundary_state(
            session,
            source_shard=self.state.shard_id,
            target_shard=target_shard,
            tensor=output,
        )
        return WindowForwardResult(tensor=output if retain_graph_tensor else output.detach(), boundary_payload=payload)
