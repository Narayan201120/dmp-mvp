import asyncio

import pytest
import torch

from daemon.main import BoundaryDeliveryError, WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.state import NodeState
from sim.network import NetworkConfig
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards


def _build_coordinator(
    *,
    network_config: NetworkConfig | None = None,
    max_staleness: int = 0,
    window_budget_ms: int = 1,
) -> tuple[WindowCoordinator, MessageBus]:
    config = ToyTransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        mlp_hidden_dim=32,
        num_layers=6,
    )
    model = build_toy_transformer(config, seed=31)
    shards, _ = build_transformer_shards(model, num_shards=3)
    bus = MessageBus()
    nodes = [
        Node(
            state=NodeState(node_id=f"node-{index}", shard_id=index),
            bus=bus,
            shard=shard,
            learning_rate=1e-2,
            snapshot_depth=8,
        )
        for index, shard in enumerate(shards)
    ]
    return (
        WindowCoordinator(
            nodes=nodes,
            network_config=network_config or NetworkConfig(),
            max_staleness=max_staleness,
            window_budget_ms=window_budget_ms,
        ),
        bus,
    )


def _training_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 4, 7, 2, 9, 3, 5, 6],
            [6, 5, 3, 9, 2, 7, 4, 1],
            [3, 1, 4, 1, 5, 9, 2, 6],
        ],
        dtype=torch.long,
    )


def test_zero_delay_training_reduces_loss_and_writes_checkpoints() -> None:
    coordinator, _ = _build_coordinator()
    batch = _training_batch()

    initial_loss = coordinator.evaluate_next_token_loss(batch)
    latest = None
    for version in range(4):
        latest = asyncio.run(coordinator.train_window(batch, version=version, microbatches=1))

    assert latest is not None
    assert latest.loss_after < latest.loss_before
    assert latest.loss_after < initial_loss
    assert set(latest.boundary_payloads) == {(0, 1), (1, 2)}

    for node in coordinator.nodes:
        assert node.state.version == 4
        assert node.state.last_checkpoint_version == 4
        assert node.snapshot_store is not None
        assert node.snapshot_store.versions() == [0, 1, 2, 3, 4]


def test_delayed_training_within_staleness_budget_still_learns() -> None:
    coordinator, bus = _build_coordinator(
        network_config=NetworkConfig(base_latency_ms=1, jitter_ms=0, packet_loss=0.0, reorder_chance=0.0),
        max_staleness=2,
        window_budget_ms=1,
    )
    batch = _training_batch()

    initial_loss = coordinator.evaluate_next_token_loss(batch)
    latest = None
    for version in range(3):
        latest = asyncio.run(coordinator.train_window(batch, version=version, microbatches=1))

    assert latest is not None
    assert latest.loss_after < latest.loss_before
    assert latest.loss_after < initial_loss
    assert bus.queue("training.boundaries").qsize() == 6

    boundary_events = []
    queue = bus.queue("training.boundaries")
    while not queue.empty():
        boundary_events.append(queue.get_nowait())

    assert all(event["status"] == "delivered" for event in boundary_events)
    assert all(event["delay_ms"] == 1 for event in boundary_events)


def test_packet_loss_fails_fast_without_advancing_state() -> None:
    coordinator, bus = _build_coordinator(
        network_config=NetworkConfig(base_latency_ms=0, jitter_ms=0, packet_loss=1.0, reorder_chance=0.0),
        max_staleness=0,
        window_budget_ms=1,
    )
    batch = _training_batch()

    with pytest.raises(BoundaryDeliveryError):
        asyncio.run(coordinator.train_window(batch, version=0, microbatches=1))

    boundary_events = []
    queue = bus.queue("training.boundaries")
    while not queue.empty():
        boundary_events.append(queue.get_nowait())

    assert len(boundary_events) == 1
    assert boundary_events[0]["status"] == "dropped"
    assert boundary_events[0]["delay_ms"] is None

    for node in coordinator.nodes:
        assert node.state.version == 0
        assert node.state.last_checkpoint_version == 0
        assert node.snapshot_store is not None
        assert node.snapshot_store.versions() == [0]
        assert node.shard is not None
        assert node.shard.training is False
