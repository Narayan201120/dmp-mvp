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
    staleness_decay_rate: float = 0.0,
    staleness_floor: float = 1.0,
    compression_topk_ratio: float = 1.0,
    compression_num_bits: int = 16,
    compression_error_feedback: bool = False,
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
            staleness_decay_rate=staleness_decay_rate,
            staleness_floor=staleness_floor,
            compression_topk_ratio=compression_topk_ratio,
            compression_num_bits=compression_num_bits,
            compression_error_feedback=compression_error_feedback,
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


def test_staleness_weighting_changes_the_training_trajectory() -> None:
    baseline_coordinator, _ = _build_coordinator()
    weighted_coordinator, weighted_bus = _build_coordinator(
        network_config=NetworkConfig(base_latency_ms=1, jitter_ms=0, packet_loss=0.0, reorder_chance=0.0),
        max_staleness=2,
        window_budget_ms=1,
        staleness_decay_rate=1.0,
        staleness_floor=0.25,
    )
    batch = _training_batch()

    baseline_latest = None
    weighted_latest = None
    for version in range(3):
        baseline_latest = asyncio.run(baseline_coordinator.train_window(batch, version=version, microbatches=1))
        weighted_latest = asyncio.run(weighted_coordinator.train_window(batch, version=version, microbatches=1))

    assert baseline_latest is not None
    assert weighted_latest is not None
    assert weighted_latest.loss_after < weighted_latest.loss_before
    assert weighted_latest.loss_after != pytest.approx(baseline_latest.loss_after)

    boundary_events = []
    queue = weighted_bus.queue("training.boundaries")
    while not queue.empty():
        boundary_events.append(queue.get_nowait())

    multipliers = [event["staleness_multiplier"] for event in boundary_events]
    assert multipliers
    assert all(multiplier < 1.0 for multiplier in multipliers)


def test_compressed_boundary_training_still_learns_and_reports_metadata() -> None:
    coordinator, bus = _build_coordinator(
        compression_topk_ratio=0.25,
        compression_num_bits=8,
    )
    batch = _training_batch()

    initial_loss = coordinator.evaluate_next_token_loss(batch)
    latest = None
    for version in range(3):
        latest = asyncio.run(coordinator.train_window(batch, version=version, microbatches=1))

    assert latest is not None
    assert latest.loss_after < latest.loss_before
    assert latest.loss_after < initial_loss

    boundary_events = []
    queue = bus.queue("training.boundaries")
    while not queue.empty():
        boundary_events.append(queue.get_nowait())

    assert boundary_events
    assert all(event["compression_applied"] for event in boundary_events)
    assert all(event["compression_num_bits"] == 8 for event in boundary_events)
    assert all(event["compressed_values"] < 3 * 7 * 16 for event in boundary_events)
    assert all(event["payload_wire_bytes"] < event["dense_payload_wire_bytes"] for event in boundary_events)
    assert all(event["payload_wire_ratio"] < 1.0 for event in boundary_events)


def test_boundary_activation_error_feedback_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported for boundary activations"):
        _build_coordinator(
            compression_topk_ratio=0.25,
            compression_num_bits=8,
            compression_error_feedback=True,
        )


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
