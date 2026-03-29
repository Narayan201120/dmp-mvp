from __future__ import annotations

from dataclasses import dataclass

import torch

from daemon.node import Node
from training.protocol import BoundaryPayload, open_window


@dataclass(frozen=True, slots=True)
class WindowRunResult:
    version: int
    logits: torch.Tensor
    boundary_payloads: dict[tuple[int, int], BoundaryPayload]


@dataclass(slots=True)
class WindowCoordinator:
    nodes: list[Node]

    async def run_window(
        self,
        input_ids: torch.Tensor,
        *,
        version: int,
        microbatches: int = 1,
    ) -> WindowRunResult:
        ordered_nodes = self._ordered_nodes()
        session = open_window(version=version, shard_count=len(ordered_nodes), microbatches=microbatches)

        hidden_states: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        for index, node in enumerate(ordered_nodes):
            if index == 0:
                output = await node.training_loop.run_window_forward(session, input_ids=input_ids)
            else:
                output = await node.training_loop.run_window_forward(session, hidden_states=hidden_states)

            if index == len(ordered_nodes) - 1:
                logits = output
            else:
                hidden_states = output

        assert logits is not None
        payloads = session.close_window()

        for node in ordered_nodes:
            node.handle_lifecycle_events(version=version + 1)

        return WindowRunResult(version=version, logits=logits, boundary_payloads=payloads)

    def _ordered_nodes(self) -> list[Node]:
        if len(self.nodes) < 2:
            raise ValueError("at least two nodes are required for a window")

        ordered = sorted(self.nodes, key=lambda node: node.state.shard_id)
        for expected_shard_id, node in enumerate(ordered):
            if node.state.shard_id != expected_shard_id:
                raise ValueError("node shard ids must be contiguous and zero-indexed")
            if node.shard is None:
                raise ValueError(f"node {node.state.node_id} is missing an attached shard")
            if node.shard.spec.shard_id != node.state.shard_id:
                raise ValueError("node shard state does not match the attached shard spec")
        return ordered
