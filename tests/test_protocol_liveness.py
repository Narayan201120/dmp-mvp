import asyncio

import torch

from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.state import NodeState
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards


async def _run_window() -> tuple[torch.Tensor, object, MessageBus, list[Node]]:
    config = ToyTransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        mlp_hidden_dim=32,
        num_layers=6,
    )
    model = build_toy_transformer(config, seed=23)
    shards, _ = build_transformer_shards(model, num_shards=3)
    bus = MessageBus()
    nodes = [
        Node(state=NodeState(node_id=f"node-{index}", shard_id=index), bus=bus, shard=shard)
        for index, shard in enumerate(shards)
    ]
    coordinator = WindowCoordinator(nodes=nodes)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        reference = model(input_ids)
        result = await coordinator.run_window(input_ids, version=0, microbatches=1)

    return reference, result, bus, nodes


def test_single_window_runtime_matches_reference_model() -> None:
    reference, result, bus, nodes = asyncio.run(_run_window())

    torch.testing.assert_close(result.logits, reference, atol=1e-6, rtol=1e-6)
    assert set(result.boundary_payloads) == {(0, 1), (1, 2)}
    assert bus.queue("training.metrics").qsize() == 3
    assert bus.queue("training.boundaries").qsize() == 2
    assert [node.state.version for node in nodes] == [1, 1, 1]
