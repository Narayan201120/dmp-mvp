from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.state import NodeState
from sim.network import NetworkConfig
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.reference import (
    build_char_vocab,
    encode_text,
    iter_eval_windows,
    load_eval_artifact,
    read_curve_rows,
    sample_batch,
    write_curve,
    write_json,
)
from training.shard import build_transformer_shards

DEFAULT_OUTPUT_CURVE = PROJECT_ROOT / "experiments" / "distributed_loss_curve.csv"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "experiments" / "distributed_run_summary.json"
DEFAULT_BASELINE_CURVE = PROJECT_ROOT / "baseline" / "baseline_loss_curve.csv"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "baseline" / "baseline_config.json"


@dataclass(slots=True)
class DistributedRunConfig:
    output_curve: str = str(DEFAULT_OUTPUT_CURVE)
    output_summary: str = str(DEFAULT_OUTPUT_SUMMARY)
    baseline_curve: str = str(DEFAULT_BASELINE_CURVE)
    baseline_summary: str = str(DEFAULT_BASELINE_SUMMARY)
    seed: int = 7
    steps: int = 100
    eval_every: int = 10
    learning_rate: float = 3e-3
    d_model: int = 32
    num_heads: int = 4
    mlp_hidden_dim: int = 64
    num_layers: int = 6
    device: str = "cpu"
    num_shards: int = 3
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-2
    snapshot_depth: int = 32
    max_staleness: int = 0
    window_budget_ms: int = 1
    staleness_decay_rate: float = 0.0
    staleness_floor: float = 1.0
    compression_topk_ratio: float = 1.0
    compression_num_bits: int = 16
    compression_error_feedback: bool = False
    base_latency_ms: int = 0
    jitter_ms: int = 0
    packet_loss: float = 0.0
    reorder_chance: float = 0.0


def parse_args() -> DistributedRunConfig:
    parser = argparse.ArgumentParser(description="Run the Phase 1 distributed comparison path.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--optimizer-name", default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--snapshot-depth", type=int, default=32)
    parser.add_argument("--max-staleness", type=int, default=0)
    parser.add_argument("--window-budget-ms", type=int, default=1)
    parser.add_argument("--staleness-decay-rate", type=float, default=0.0)
    parser.add_argument("--staleness-floor", type=float, default=1.0)
    parser.add_argument("--compression-topk-ratio", type=float, default=1.0)
    parser.add_argument("--compression-num-bits", type=int, default=16)
    parser.add_argument("--compression-error-feedback", action="store_true")
    parser.add_argument("--base-latency-ms", type=int, default=0)
    parser.add_argument("--jitter-ms", type=int, default=0)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--reorder-chance", type=float, default=0.0)
    parser.add_argument("--output-curve", default=str(DEFAULT_OUTPUT_CURVE))
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--baseline-curve", default=str(DEFAULT_BASELINE_CURVE))
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    args = parser.parse_args()
    return DistributedRunConfig(
        output_curve=args.output_curve,
        output_summary=args.output_summary,
        baseline_curve=args.baseline_curve,
        baseline_summary=args.baseline_summary,
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
        num_shards=args.num_shards,
        optimizer_name=args.optimizer_name,
        weight_decay=args.weight_decay,
        snapshot_depth=args.snapshot_depth,
        max_staleness=args.max_staleness,
        window_budget_ms=args.window_budget_ms,
        staleness_decay_rate=args.staleness_decay_rate,
        staleness_floor=args.staleness_floor,
        compression_topk_ratio=args.compression_topk_ratio,
        compression_num_bits=args.compression_num_bits,
        compression_error_feedback=args.compression_error_feedback,
        base_latency_ms=args.base_latency_ms,
        jitter_ms=args.jitter_ms,
        packet_loss=args.packet_loss,
        reorder_chance=args.reorder_chance,
    )


def evaluate(coordinator: WindowCoordinator, eval_windows: list[torch.Tensor]) -> float:
    losses = [coordinator.evaluate_next_token_loss(window) for window in eval_windows]
    return sum(losses) / len(losses)


def run(config: DistributedRunConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    artifact = load_eval_artifact()
    stoi, _ = build_char_vocab(artifact.text)
    encoded = encode_text(artifact.text, stoi)
    device = torch.device(config.device)
    sequence_length = min(artifact.seq_len, encoded.size(0) - 1)
    eval_windows = iter_eval_windows(encoded, sequence_length=sequence_length, device=device)

    baseline_reference = load_baseline_reference(config)
    coordinator = build_coordinator(config, vocab_size=len(stoi), sequence_length=sequence_length, device=device)

    rows, final_train_loss = asyncio.run(
        train_distributed(
            coordinator,
            config=config,
            encoded=encoded,
            batch_size=artifact.batch_size,
            sequence_length=sequence_length,
            eval_windows=eval_windows,
            device=device,
        )
    )

    output_curve = Path(config.output_curve)
    write_curve(output_curve, rows)
    final_eval_loss = float(rows[-1]["eval_loss"])
    boundary_summary = summarize_boundary_events(coordinator.nodes[0].bus.queue("training.boundaries"))
    final_version = max(node.state.version for node in coordinator.nodes)

    summary = {
        "config": asdict(config),
        "dataset": "shakespeare_public_domain_excerpt_v1",
        "vocab_size": len(stoi),
        "sequence_length": sequence_length,
        "batch_size": artifact.batch_size,
        "initial_eval_loss": float(rows[0]["eval_loss"]),
        "final_train_loss": float(final_train_loss),
        "final_eval_loss": final_eval_loss,
        "final_version": final_version,
        "curve_path": str(output_curve),
        "boundary_events_recorded": boundary_summary["events_recorded"],
        "boundary_delivery": boundary_summary,
        "baseline_reference": {
            "curve_path": str(Path(config.baseline_curve)),
            "summary_path": str(Path(config.baseline_summary)),
            "final_eval_loss": baseline_reference["final_eval_loss"],
            "final_eval_loss_delta": final_eval_loss - baseline_reference["final_eval_loss"],
            "eval_loss_ratio": final_eval_loss / baseline_reference["final_eval_loss"],
            "schedule_matches": True,
            "config_matches": True,
        },
    }
    write_json(Path(config.output_summary), summary)
    return summary


def summarize_boundary_events(queue) -> dict[str, object]:
    events: list[dict[str, object]] = []
    while not queue.empty():
        events.append(queue.get_nowait())

    status_counts = Counter(str(event.get("status", "unknown")) for event in events)
    delay_values = [int(delay) for delay in (event.get("delay_ms") for event in events) if delay is not None]
    simulated_versions = [
        int(version)
        for version in (event.get("simulated_current_version") for event in events)
        if version is not None
    ]
    staleness_multipliers = [
        float(multiplier)
        for multiplier in (event.get("staleness_multiplier") for event in events)
        if multiplier is not None
    ]
    compressed_value_counts = [
        int(count)
        for count in (event.get("compressed_values") for event in events)
        if count is not None
    ]
    compression_applied_count = sum(1 for event in events if bool(event.get("compression_applied")))

    return {
        "events_recorded": len(events),
        "status_counts": dict(status_counts),
        "delay_ms": {
            "min": min(delay_values) if delay_values else None,
            "max": max(delay_values) if delay_values else None,
            "avg": (sum(delay_values) / len(delay_values)) if delay_values else None,
        },
        "max_simulated_current_version": max(simulated_versions) if simulated_versions else None,
        "staleness_multiplier": {
            "min": min(staleness_multipliers) if staleness_multipliers else None,
            "max": max(staleness_multipliers) if staleness_multipliers else None,
            "avg": (sum(staleness_multipliers) / len(staleness_multipliers)) if staleness_multipliers else None,
        },
        "compression": {
            "applied_events": compression_applied_count,
            "min_values": min(compressed_value_counts) if compressed_value_counts else None,
            "max_values": max(compressed_value_counts) if compressed_value_counts else None,
            "avg_values": (sum(compressed_value_counts) / len(compressed_value_counts))
            if compressed_value_counts
            else None,
        },
    }


def load_baseline_reference(config: DistributedRunConfig) -> dict[str, float]:
    summary_path = Path(config.baseline_summary)
    curve_path = Path(config.baseline_curve)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve_rows = read_curve_rows(curve_path)
    if not curve_rows:
        raise RuntimeError("baseline curve is empty")

    baseline_config = summary.get("config", {})
    mismatches: list[str] = []
    for field_name in (
        "steps",
        "eval_every",
        "seed",
        "learning_rate",
        "d_model",
        "num_heads",
        "mlp_hidden_dim",
        "num_layers",
        "device",
    ):
        expected = getattr(config, field_name)
        actual = baseline_config.get(field_name)
        if actual != expected:
            mismatches.append(f"{field_name}: expected {expected!r}, found {actual!r}")

    if summary.get("dataset") != "shakespeare_public_domain_excerpt_v1":
        mismatches.append("dataset does not match the frozen Phase 0 reference")

    final_curve_eval_loss = float(curve_rows[-1]["eval_loss"])
    final_summary_eval_loss = float(summary["final_eval_loss"])
    if abs(final_curve_eval_loss - final_summary_eval_loss) > 1e-9:
        raise RuntimeError("baseline curve and baseline summary disagree on final_eval_loss")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"distributed comparison config does not match the frozen baseline: {mismatch_text}")

    return {"final_eval_loss": final_summary_eval_loss}


def build_coordinator(
    config: DistributedRunConfig,
    *,
    vocab_size: int,
    sequence_length: int,
    device: torch.device,
) -> WindowCoordinator:
    model = build_toy_transformer(
        ToyTransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=sequence_length,
            d_model=config.d_model,
            num_heads=config.num_heads,
            mlp_hidden_dim=config.mlp_hidden_dim,
            num_layers=config.num_layers,
        ),
        seed=config.seed,
        device=device,
    )
    shards, _ = build_transformer_shards(model, num_shards=config.num_shards)
    bus = MessageBus()
    nodes = [
        Node(
            state=NodeState(node_id=f"node-{index}", shard_id=index),
            bus=bus,
            shard=shard,
            learning_rate=config.learning_rate,
            optimizer_name=config.optimizer_name,
            weight_decay=config.weight_decay,
            snapshot_depth=config.snapshot_depth,
        )
        for index, shard in enumerate(shards)
    ]
    return WindowCoordinator(
        nodes=nodes,
        network_config=NetworkConfig(
            base_latency_ms=config.base_latency_ms,
            jitter_ms=config.jitter_ms,
            packet_loss=config.packet_loss,
            reorder_chance=config.reorder_chance,
        ),
        max_staleness=config.max_staleness,
        window_budget_ms=config.window_budget_ms,
        staleness_decay_rate=config.staleness_decay_rate,
        staleness_floor=config.staleness_floor,
        compression_topk_ratio=config.compression_topk_ratio,
        compression_num_bits=config.compression_num_bits,
        compression_error_feedback=config.compression_error_feedback,
        rng=random.Random(config.seed + 1),
    )


async def train_distributed(
    coordinator: WindowCoordinator,
    *,
    config: DistributedRunConfig,
    encoded: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    eval_windows: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[dict[str, float]], float]:
    rng = random.Random(config.seed)
    rows: list[dict[str, float]] = []
    initial_eval_loss = evaluate(coordinator, eval_windows)
    rows.append({"step": 0, "train_loss": initial_eval_loss, "eval_loss": initial_eval_loss})

    train_loss = float(initial_eval_loss)
    for step in range(1, config.steps + 1):
        batch = sample_batch(
            encoded,
            batch_size=batch_size,
            sequence_length=sequence_length,
            rng=rng,
            device=device,
        )
        result = await coordinator.train_window(batch, version=step - 1, microbatches=1)
        train_loss = float(result.loss_before)

        if step % config.eval_every == 0 or step == config.steps:
            eval_loss = evaluate(coordinator, eval_windows)
            rows.append({"step": step, "train_loss": train_loss, "eval_loss": eval_loss})

    return rows, train_loss


def main() -> None:
    config = parse_args()
    summary = run(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
