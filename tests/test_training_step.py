import asyncio

import torch

from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.state import NodeState
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards


def _build_coordinator() -> WindowCoordinator:
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
    return WindowCoordinator(nodes=nodes)


def test_zero_delay_training_reduces_loss_and_writes_checkpoints() -> None:
    coordinator = _build_coordinator()
    batch = torch.tensor(
        [
            [1, 4, 7, 2, 9, 3, 5, 6],
            [6, 5, 3, 9, 2, 7, 4, 1],
            [3, 1, 4, 1, 5, 9, 2, 6],
        ],
        dtype=torch.long,
    )

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
