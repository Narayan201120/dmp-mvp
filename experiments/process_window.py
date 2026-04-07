from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.process_runtime import ProcessWindowRunner
from daemon.state import NodeState
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.reference import write_json
from training.shard import build_transformer_shards

DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "experiments" / "process_window_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-process shard smoke test for forward and one train step.")
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--num-shards", type=int, default=3)
    return parser.parse_args()


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


def run(*, output_summary: Path, num_shards: int) -> dict[str, object]:
    config = ToyTransformerConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        mlp_hidden_dim=32,
        num_layers=6,
    )
    model = build_toy_transformer(config, seed=23)
    process_shards, _ = build_transformer_shards(model, num_shards=num_shards)
    reference_shards, _ = build_transformer_shards(model, num_shards=num_shards)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    training_batch = _training_batch()

    with torch.no_grad():
        reference = model(input_ids).to(dtype=torch.float32)

    reference_train = asyncio.run(_build_reference_coordinator(reference_shards).train_window(training_batch, version=0))

    with ProcessWindowRunner(process_shards, learning_rate=1e-2, optimizer_name="adam") as runner:
        result = runner.run_window(input_ids, version=0)
        process_train = runner.train_window(training_batch, version=0)
        process_status = runner.process_status()

    max_abs_diff = float(torch.max(torch.abs(reference - result)).item())
    loss_before_delta = abs(process_train.loss_before - reference_train.loss_before)
    loss_after_delta = abs(process_train.loss_after - reference_train.loss_after)
    summary = {
        "num_shards": num_shards,
        "input_shape": list(input_ids.shape),
        "logits_shape": list(result.shape),
        "max_abs_diff": max_abs_diff,
        "matches_reference": bool(torch.allclose(reference, result, atol=1e-6, rtol=1e-6)),
        "train_step": {
            "loss_before": process_train.loss_before,
            "loss_after": process_train.loss_after,
            "reference_loss_before": reference_train.loss_before,
            "reference_loss_after": reference_train.loss_after,
            "loss_before_delta": loss_before_delta,
            "loss_after_delta": loss_after_delta,
            "matches_reference": loss_before_delta <= 1e-6 and loss_after_delta <= 1e-6,
        },
        "processes": process_status,
    }
    write_json(output_summary, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(output_summary=Path(args.output_summary), num_shards=args.num_shards)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
