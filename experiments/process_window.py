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
from daemon.process_runtime import ProcessWindowRunner, ProcessWorkerEndpoint
from daemon.state import NodeState
from sim.network import NetworkConfig
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.reference import write_json
from training.shard import build_transformer_shards

DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "experiments" / "process_window_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-process shard smoke test for forward and one train step.")
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--worker-endpoints", type=str, default="")
    parser.add_argument("--stop-workers-on-close", action="store_true")
    parser.add_argument("--print-launch-plan", action="store_true")
    parser.add_argument("--base-latency-ms", type=int, default=0)
    parser.add_argument("--jitter-ms", type=int, default=0)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--reorder-chance", type=float, default=0.0)
    parser.add_argument("--max-staleness", type=int, default=0)
    parser.add_argument("--window-budget-ms", type=int, default=1)
    parser.add_argument("--staleness-decay-rate", type=float, default=0.0)
    parser.add_argument("--staleness-floor", type=float, default=1.0)
    parser.add_argument("--compression-topk-ratio", type=float, default=1.0)
    parser.add_argument("--compression-num-bits", type=int, default=16)
    return parser.parse_args()


def _parse_worker_endpoints(raw_value: str) -> list[ProcessWorkerEndpoint] | None:
    if not raw_value.strip():
        return None
    endpoints: list[ProcessWorkerEndpoint] = []
    for item in raw_value.split(","):
        host, port = item.strip().rsplit(":", maxsplit=1)
        endpoints.append(ProcessWorkerEndpoint(host=host, port=int(port)))
    return endpoints


def build_external_worker_launch_plan(
    *,
    worker_endpoints: list[ProcessWorkerEndpoint],
    compression_topk_ratio: float,
    compression_num_bits: int,
    base_latency_ms: int,
    jitter_ms: int,
    packet_loss: float,
    reorder_chance: float,
    max_staleness: int,
    window_budget_ms: int,
    staleness_decay_rate: float,
    staleness_floor: float,
    num_shards: int,
    python_executable: str = "python",
) -> dict[str, object]:
    if not worker_endpoints:
        raise ValueError("worker_endpoints are required to build a launch plan")

    endpoint_arg = ",".join(f"{endpoint.host}:{endpoint.port}" for endpoint in worker_endpoints)
    return {
        "worker_commands": [
            [
                python_executable,
                "-m",
                "daemon.socket_worker",
                "--bind-host",
                "0.0.0.0",
                "--port",
                str(endpoint.port),
            ]
            for endpoint in worker_endpoints
        ],
        "coordinator_command": [
            python_executable,
            "experiments/process_window.py",
            "--num-shards",
            str(num_shards),
            "--worker-endpoints",
            endpoint_arg,
            "--compression-topk-ratio",
            str(compression_topk_ratio),
            "--compression-num-bits",
            str(compression_num_bits),
            "--base-latency-ms",
            str(base_latency_ms),
            "--jitter-ms",
            str(jitter_ms),
            "--packet-loss",
            str(packet_loss),
            "--reorder-chance",
            str(reorder_chance),
            "--max-staleness",
            str(max_staleness),
            "--window-budget-ms",
            str(window_budget_ms),
            "--staleness-decay-rate",
            str(staleness_decay_rate),
            "--staleness-floor",
            str(staleness_floor),
        ],
    }


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


def run(
    *,
    output_summary: Path,
    num_shards: int,
    worker_endpoints: list[ProcessWorkerEndpoint] | None = None,
    stop_workers_on_close: bool | None = None,
    network_config: NetworkConfig | None = None,
    max_staleness: int = 0,
    window_budget_ms: int = 1,
    staleness_decay_rate: float = 0.0,
    staleness_floor: float = 1.0,
    compression_topk_ratio: float = 1.0,
    compression_num_bits: int = 16,
) -> dict[str, object]:
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

    reference_coordinator = _build_reference_coordinator(reference_shards)
    reference_train_results = [
        asyncio.run(reference_coordinator.train_window(training_batch, version=version))
        for version in range(3)
    ]

    with ProcessWindowRunner(
        process_shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        worker_endpoints=worker_endpoints,
        stop_workers_on_close=stop_workers_on_close,
        network_config=network_config or NetworkConfig(),
        max_staleness=max_staleness,
        window_budget_ms=window_budget_ms,
        staleness_decay_rate=staleness_decay_rate,
        staleness_floor=staleness_floor,
        compression_topk_ratio=compression_topk_ratio,
        compression_num_bits=compression_num_bits,
    ) as runner:
        result = runner.run_window(input_ids, version=0)
        process_train_results = [runner.train_window(training_batch, version=version) for version in range(3)]
        process_status = runner.process_status()
        boundary_events = runner.last_boundary_events()

    max_abs_diff = float(torch.max(torch.abs(reference - result)).item())
    final_process_train = process_train_results[-1]
    final_reference_train = reference_train_results[-1]
    loss_before_delta = abs(final_process_train.loss_before - final_reference_train.loss_before)
    loss_after_delta = abs(final_process_train.loss_after - final_reference_train.loss_after)
    summary = {
        "num_shards": num_shards,
        "worker_mode": "external" if worker_endpoints is not None else "managed_local",
        "transport_config": {
            "base_latency_ms": (network_config or NetworkConfig()).base_latency_ms,
            "jitter_ms": (network_config or NetworkConfig()).jitter_ms,
            "packet_loss": (network_config or NetworkConfig()).packet_loss,
            "reorder_chance": (network_config or NetworkConfig()).reorder_chance,
            "max_staleness": max_staleness,
            "window_budget_ms": window_budget_ms,
            "staleness_decay_rate": staleness_decay_rate,
            "staleness_floor": staleness_floor,
            "compression_topk_ratio": compression_topk_ratio,
            "compression_num_bits": compression_num_bits,
        },
        "input_shape": list(input_ids.shape),
        "logits_shape": list(result.shape),
        "max_abs_diff": max_abs_diff,
        "matches_reference": bool(torch.allclose(reference, result, atol=1e-6, rtol=1e-6)),
        "train_step": {
            "num_steps": 3,
            "loss_before": final_process_train.loss_before,
            "loss_after": final_process_train.loss_after,
            "reference_loss_before": final_reference_train.loss_before,
            "reference_loss_after": final_reference_train.loss_after,
            "loss_before_delta": loss_before_delta,
            "loss_after_delta": loss_after_delta,
            "matches_reference": loss_before_delta <= 1e-6 and loss_after_delta <= 1e-6,
            "loss_history": [
                {
                    "version": process_result.version,
                    "loss_before": process_result.loss_before,
                    "loss_after": process_result.loss_after,
                    "reference_loss_before": reference_result.loss_before,
                    "reference_loss_after": reference_result.loss_after,
                }
                for process_result, reference_result in zip(process_train_results, reference_train_results)
            ],
        },
        "boundary_events": boundary_events,
        "processes": process_status,
    }
    write_json(output_summary, summary)
    return summary


def main() -> None:
    args = parse_args()
    worker_endpoints = _parse_worker_endpoints(args.worker_endpoints)
    if args.print_launch_plan:
        if worker_endpoints is None:
            raise ValueError("--print-launch-plan requires --worker-endpoints")
        plan = build_external_worker_launch_plan(
            worker_endpoints=worker_endpoints,
            compression_topk_ratio=args.compression_topk_ratio,
            compression_num_bits=args.compression_num_bits,
            base_latency_ms=args.base_latency_ms,
            jitter_ms=args.jitter_ms,
            packet_loss=args.packet_loss,
            reorder_chance=args.reorder_chance,
            max_staleness=args.max_staleness,
            window_budget_ms=args.window_budget_ms,
            staleness_decay_rate=args.staleness_decay_rate,
            staleness_floor=args.staleness_floor,
            num_shards=args.num_shards,
        )
        print(json.dumps(plan, indent=2))
        return
    summary = run(
        output_summary=Path(args.output_summary),
        num_shards=args.num_shards,
        worker_endpoints=worker_endpoints,
        stop_workers_on_close=args.stop_workers_on_close,
        network_config=NetworkConfig(
            base_latency_ms=args.base_latency_ms,
            jitter_ms=args.jitter_ms,
            packet_loss=args.packet_loss,
            reorder_chance=args.reorder_chance,
        ),
        max_staleness=args.max_staleness,
        window_budget_ms=args.window_budget_ms,
        staleness_decay_rate=args.staleness_decay_rate,
        staleness_floor=args.staleness_floor,
        compression_topk_ratio=args.compression_topk_ratio,
        compression_num_bits=args.compression_num_bits,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
