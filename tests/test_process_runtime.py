import asyncio
import json
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.process_runtime import ProcessWindowRunner
from daemon.state import NodeState
from experiments.process_window import run
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards


def _make_workspace_tmp_dir() -> Path:
    root = Path.cwd() / "tests_runtime_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    return path


def _training_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [1, 4, 7, 2, 9, 3, 5, 6],
            [6, 5, 3, 9, 2, 7, 4, 1],
            [3, 1, 4, 1, 5, 9, 2, 6],
        ],
        dtype=torch.long,
    )


def _build_reference_coordinator(shards) -> WindowCoordinator:
    bus = MessageBus()
    nodes = [
        Node(
            state=NodeState(node_id=f"node-{index}", shard_id=index),
            bus=bus,
            shard=shard,
            learning_rate=1e-2,
            optimizer_name="adam",
            snapshot_depth=8,
        )
        for index, shard in enumerate(shards)
    ]
    return WindowCoordinator(nodes=nodes)


def test_process_window_runner_matches_reference_model() -> None:
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
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        reference = model(input_ids).to(dtype=torch.float32)

    with ProcessWindowRunner(shards) as runner:
        result = runner.run_window(input_ids, version=0)
        statuses = runner.process_status()

    torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-6)
    assert len(statuses) == 3
    assert all(status["alive"] is True for status in statuses)


def test_process_window_training_matches_in_process_coordinator() -> None:
    config = ToyTransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        mlp_hidden_dim=32,
        num_layers=6,
    )
    model = build_toy_transformer(config, seed=23)
    process_shards, _ = build_transformer_shards(model, num_shards=3)
    reference_shards, _ = build_transformer_shards(model, num_shards=3)
    training_batch = _training_batch()

    reference_result = asyncio.run(_build_reference_coordinator(reference_shards).train_window(training_batch, version=0))

    with ProcessWindowRunner(process_shards, learning_rate=1e-2, optimizer_name="adam") as runner:
        process_result = runner.train_window(training_batch, version=0)

    assert process_result.loss_before == reference_result.loss_before
    assert process_result.loss_after == reference_result.loss_after
    assert process_result.loss_after < process_result.loss_before


def test_process_window_experiment_writes_summary() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        output_summary = tmp_dir / "process_window_summary.json"
        summary = run(output_summary=output_summary, num_shards=3)

        written = json.loads(output_summary.read_text(encoding="utf-8"))
        assert summary["matches_reference"] is True
        assert summary["max_abs_diff"] == 0.0
        assert summary["train_step"]["matches_reference"] is True
        assert written["matches_reference"] is True
        assert written["train_step"]["matches_reference"] is True
        assert written["num_shards"] == 3
        assert len(written["processes"]) == 3
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
