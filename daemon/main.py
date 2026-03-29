from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import random

import torch

from daemon.node import Node
from sim.network import NetworkConfig, sample_delivery_delay, wrap_message
from training.metrics import next_token_loss, split_next_token_batch
from training.protocol import BoundaryPayload, open_window, payload_as_tensor
from training.staleness import should_drop


class BoundaryDeliveryError(RuntimeError):
    """Raised when a boundary payload cannot be delivered."""


class StaleBoundaryError(BoundaryDeliveryError):
    """Raised when a boundary payload arrives too late for the configured staleness budget."""


@dataclass(frozen=True, slots=True)
class WindowRunResult:
    version: int
    logits: torch.Tensor
    boundary_payloads: dict[tuple[int, int], BoundaryPayload]


@dataclass(frozen=True, slots=True)
class TrainWindowResult:
    version: int
    loss_before: float
    loss_after: float
    boundary_payloads: dict[tuple[int, int], BoundaryPayload]


@dataclass(slots=True)
class WindowCoordinator:
    nodes: list[Node]
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    max_staleness: int = 0
    window_budget_ms: int = 1
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def __post_init__(self) -> None:
        if self.max_staleness < 0:
            raise ValueError("max_staleness must be non-negative")
        if self.window_budget_ms < 1:
            raise ValueError("window_budget_ms must be at least 1")

    async def run_window(
        self,
        input_ids: torch.Tensor,
        *,
        version: int,
        microbatches: int = 1,
    ) -> WindowRunResult:
        ordered_nodes, logits, payloads = await self._execute_window(
            input_ids,
            version=version,
            microbatches=microbatches,
            retain_graph_tensor=False,
        )

        for node in ordered_nodes:
            node.handle_lifecycle_events(version=version + 1)

        return WindowRunResult(version=version, logits=logits, boundary_payloads=payloads)

    async def train_window(
        self,
        input_ids: torch.Tensor,
        *,
        version: int,
        microbatches: int = 1,
    ) -> TrainWindowResult:
        model_inputs, targets = split_next_token_batch(input_ids)
        ordered_nodes = self._ordered_nodes()
        loss_before = self.evaluate_next_token_loss(input_ids)

        try:
            for node in ordered_nodes:
                node.set_training_mode(True)
                node.zero_grad()

            ordered_nodes, logits, payloads = await self._execute_window(
                model_inputs,
                version=version,
                microbatches=microbatches,
                retain_graph_tensor=True,
            )
            loss = next_token_loss(logits, targets)
            loss.backward()

            for node in ordered_nodes:
                node.optimizer_step()
                node.save_checkpoint(version + 1)
                node.handle_lifecycle_events(version=version + 1)

            loss_after = self.evaluate_next_token_loss(input_ids)
            return TrainWindowResult(
                version=version,
                loss_before=float(loss_before),
                loss_after=float(loss_after),
                boundary_payloads=payloads,
            )
        finally:
            self._reset_training_state(ordered_nodes)

    def evaluate_next_token_loss(self, input_ids: torch.Tensor) -> float:
        model_inputs, targets = split_next_token_batch(input_ids)
        ordered_nodes = self._ordered_nodes()
        for node in ordered_nodes:
            node.set_training_mode(False)
        with torch.no_grad():
            logits = self._direct_forward(ordered_nodes, model_inputs)
            loss = next_token_loss(logits, targets)
        return float(loss.item())

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

    async def _execute_window(
        self,
        input_ids: torch.Tensor,
        *,
        version: int,
        microbatches: int,
        retain_graph_tensor: bool,
    ) -> tuple[list[Node], torch.Tensor, dict[tuple[int, int], BoundaryPayload]]:
        ordered_nodes = self._ordered_nodes()
        session = open_window(version=version, shard_count=len(ordered_nodes), microbatches=microbatches)

        hidden_states: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        for index, node in enumerate(ordered_nodes):
            if index == 0:
                result = await node.training_loop.run_window_forward(
                    session,
                    input_ids=input_ids,
                    retain_graph_tensor=retain_graph_tensor,
                )
            else:
                result = await node.training_loop.run_window_forward(
                    session,
                    hidden_states=hidden_states,
                    retain_graph_tensor=retain_graph_tensor,
                )

            if index == len(ordered_nodes) - 1:
                logits = result.tensor
            else:
                if result.boundary_payload is None:
                    raise RuntimeError("intermediate shard did not emit a boundary payload")
                hidden_states = await self._deliver_boundary(
                    session=session,
                    source_node=node,
                    payload=result.boundary_payload,
                    source_tensor=result.tensor,
                    retain_graph_tensor=retain_graph_tensor,
                )

        assert logits is not None
        payloads = session.close_window()
        return ordered_nodes, logits, payloads

    def _direct_forward(self, ordered_nodes: list[Node], input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states: torch.Tensor | None = None
        logits: torch.Tensor | None = None

        for index, node in enumerate(ordered_nodes):
            if node.shard is None:
                raise RuntimeError("node is missing an attached shard")
            if index == 0:
                output = node.shard(input_ids=input_ids)
            else:
                output = node.shard(hidden_states=hidden_states)

            if index == len(ordered_nodes) - 1:
                logits = output
            else:
                hidden_states = output

        if logits is None:
            raise RuntimeError("direct forward did not produce logits")
        return logits

    def _reset_training_state(self, ordered_nodes: list[Node]) -> None:
        for node in ordered_nodes:
            node.zero_grad()
            node.set_training_mode(False)

    async def _deliver_boundary(
        self,
        *,
        session,
        source_node: Node,
        payload: BoundaryPayload,
        source_tensor: torch.Tensor,
        retain_graph_tensor: bool,
    ) -> torch.Tensor:
        delay_ms = sample_delivery_delay(self.network_config, self.rng)
        message = wrap_message(
            {
                "version": payload.version,
                "source_shard": payload.source_shard,
                "target_shard": payload.target_shard,
                "shape": tuple(payload.tensor.shape),
            },
            delay_ms=delay_ms,
        )

        if delay_ms is None:
            await source_node.bus.publish("training.boundaries", {**message, "status": "dropped"})
            raise BoundaryDeliveryError(
                f"boundary {payload.source_shard}->{payload.target_shard} was dropped by the network"
            )

        if delay_ms:
            await asyncio.sleep(delay_ms / 1000.0)

        simulated_current_version = payload.version + math.ceil(delay_ms / self.window_budget_ms)
        if should_drop(payload.version, simulated_current_version, self.max_staleness):
            await source_node.bus.publish(
                "training.boundaries",
                {
                    **message,
                    "status": "stale",
                    "simulated_current_version": simulated_current_version,
                },
            )
            raise StaleBoundaryError(
                f"boundary {payload.source_shard}->{payload.target_shard} became stale before delivery"
            )

        session.exchange_boundary_state(payload)
        await source_node.bus.publish(
            "training.boundaries",
            {
                **message,
                "status": "delivered",
                "simulated_current_version": simulated_current_version,
            },
        )

        if retain_graph_tensor:
            return source_tensor
        return payload_as_tensor(payload, device=source_tensor.device, dtype=source_tensor.dtype)
