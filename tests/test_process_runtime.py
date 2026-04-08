import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from uuid import uuid4

import pytest
import torch

from daemon.main import StaleBoundaryError
from daemon.main import WindowCoordinator
from daemon.messaging import MessageBus
from daemon.node import Node
from daemon.process_runtime import ProcessWindowRunner, ProcessWorkerEndpoint
from daemon.state import NodeState
from experiments.process_window import run
from sim.network import NetworkConfig
from training.model_factory import ToyTransformerConfig, build_toy_transformer
from training.shard import build_transformer_shards


def _make_workspace_tmp_dir() -> Path:
    root = Path.cwd() / "tests_runtime_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    return path


def _reserve_worker_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _spawn_external_workers(num_workers: int) -> tuple[list[ProcessWorkerEndpoint], list[subprocess.Popen[bytes]]]:
    endpoints = [ProcessWorkerEndpoint(host="127.0.0.1", port=_reserve_worker_port()) for _ in range(num_workers)]
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for endpoint in endpoints:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "daemon.socket_worker",
                        "--bind-host",
                        endpoint.host,
                        "--port",
                        str(endpoint.port),
                    ],
                    cwd=str(Path.cwd()),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )
        yield endpoints, processes
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


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

    reference_coordinator = _build_reference_coordinator(reference_shards)
    reference_results = []
    for version in range(3):
        reference_results.append(asyncio.run(reference_coordinator.train_window(training_batch, version=version)))

    with ProcessWindowRunner(
        process_shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        snapshot_depth=8,
    ) as runner:
        process_results = [runner.train_window(training_batch, version=version) for version in range(3)]
        statuses = runner.process_status()

    for process_result, reference_result in zip(process_results, reference_results):
        assert process_result.loss_before == reference_result.loss_before
        assert process_result.loss_after == reference_result.loss_after
        assert process_result.loss_after < process_result.loss_before

    assert len(statuses) == 3
    assert all(status["alive"] is True for status in statuses)
    assert all(status["version"] == 3 for status in statuses)
    assert all(status["last_checkpoint_version"] == 3 for status in statuses)
    assert all(status["checkpoint_versions"] == [0, 1, 2, 3] for status in statuses)


def test_process_window_attaches_to_external_workers() -> None:
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
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
        ],
        dtype=torch.long,
    )
    training_batch = _training_batch()

    with torch.no_grad():
        reference_logits = model(input_ids).to(dtype=torch.float32)

    reference_result = asyncio.run(_build_reference_coordinator(reference_shards).train_window(training_batch, version=0))

    with _spawn_external_workers(3) as (endpoints, processes):
        with ProcessWindowRunner(
            process_shards,
            learning_rate=1e-2,
            optimizer_name="adam",
            snapshot_depth=8,
            worker_endpoints=endpoints,
            stop_workers_on_close=True,
        ) as runner:
            logits = runner.run_window(input_ids, version=0)
            process_result = runner.train_window(training_batch, version=0)
            statuses = runner.process_status()

        for process in processes:
            process.wait(timeout=10)

    torch.testing.assert_close(logits, reference_logits, atol=1e-6, rtol=1e-6)
    assert process_result.loss_before == reference_result.loss_before
    assert process_result.loss_after == reference_result.loss_after
    assert all(status["managed"] is False for status in statuses)
    assert all(status["alive"] is True for status in statuses)
    assert all(status["version"] == 1 for status in statuses)


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
        assert all(process["version"] == 3 for process in written["processes"])
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_process_window_compression_reports_boundary_metadata() -> None:
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
    training_batch = _training_batch()

    with ProcessWindowRunner(
        shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        compression_topk_ratio=0.25,
        compression_num_bits=8,
    ) as runner:
        result = runner.train_window(training_batch, version=0)
        boundary_events = runner.last_boundary_events()
        statuses = runner.process_status()

    assert result.loss_after < result.loss_before
    assert len(boundary_events) == 2
    assert all(event["compression_applied"] is True for event in boundary_events)
    assert all(event["compression_num_bits"] == 8 for event in boundary_events)
    assert all(event["payload_wire_bytes"] < event["dense_payload_wire_bytes"] for event in boundary_events)
    assert all(event["payload_wire_ratio"] < 1.0 for event in boundary_events)
    assert all(status["version"] == 1 for status in statuses)
    assert all(status["checkpoint_versions"] == [0, 1] for status in statuses)


def test_process_window_stale_boundary_rejects_before_commit() -> None:
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
    training_batch = _training_batch()

    with ProcessWindowRunner(
        shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        network_config=NetworkConfig(base_latency_ms=1, jitter_ms=0, packet_loss=0.0, reorder_chance=0.0),
        max_staleness=0,
        window_budget_ms=1,
    ) as runner:
        with pytest.raises(StaleBoundaryError):
            runner.train_window(training_batch, version=0)
        boundary_events = runner.last_boundary_events()
        statuses = runner.process_status()

    assert len(boundary_events) == 1
    assert boundary_events[0]["status"] == "stale"
    assert all(status["version"] == 0 for status in statuses)
    assert all(status["checkpoint_versions"] == [0] for status in statuses)


def test_process_window_partial_commit_failure_rolls_back_all_workers(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with ProcessWindowRunner(
        process_shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        snapshot_depth=8,
    ) as runner:
        original_request_worker = ProcessWindowRunner._request_worker

        def flaky_request_worker(self: ProcessWindowRunner, index: int, request: dict[str, object]) -> dict[str, object]:
            if request["kind"] == "commit" and index == 1:
                raise RuntimeError("synthetic commit failure")
            return original_request_worker(self, index, request)

        monkeypatch.setattr(ProcessWindowRunner, "_request_worker", flaky_request_worker)
        with pytest.raises(RuntimeError, match="synthetic commit failure"):
            runner.train_window(training_batch, version=0)
        failed_statuses = runner.process_status()

        monkeypatch.setattr(ProcessWindowRunner, "_request_worker", original_request_worker)
        recovered_result = runner.train_window(training_batch, version=0)
        recovered_statuses = runner.process_status()

    assert all(status["version"] == 0 for status in failed_statuses)
    assert all(status["checkpoint_versions"] == [0] for status in failed_statuses)
    assert recovered_result.loss_before == reference_result.loss_before
    assert recovered_result.loss_after == reference_result.loss_after
    assert all(status["version"] == 1 for status in recovered_statuses)
    assert all(status["checkpoint_versions"] == [0, 1] for status in recovered_statuses)


def test_process_window_partial_commit_failure_at_later_version_rolls_back_to_last_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    reference_coordinator = _build_reference_coordinator(reference_shards)
    _ = asyncio.run(reference_coordinator.train_window(training_batch, version=0))
    reference_result = asyncio.run(reference_coordinator.train_window(training_batch, version=1))

    with ProcessWindowRunner(
        process_shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        snapshot_depth=8,
    ) as runner:
        _ = runner.train_window(training_batch, version=0)
        original_request_worker = ProcessWindowRunner._request_worker

        def flaky_request_worker(self: ProcessWindowRunner, index: int, request: dict[str, object]) -> dict[str, object]:
            if request["kind"] == "commit" and request["version"] == 1 and index == 1:
                raise RuntimeError("synthetic later commit failure")
            return original_request_worker(self, index, request)

        monkeypatch.setattr(ProcessWindowRunner, "_request_worker", flaky_request_worker)
        with pytest.raises(RuntimeError, match="synthetic later commit failure"):
            runner.train_window(training_batch, version=1)
        failed_statuses = runner.process_status()

        monkeypatch.setattr(ProcessWindowRunner, "_request_worker", original_request_worker)
        recovered_result = runner.train_window(training_batch, version=1)
        recovered_statuses = runner.process_status()

    assert all(status["version"] == 1 for status in failed_statuses)
    assert all(status["checkpoint_versions"] == [0, 1] for status in failed_statuses)
    assert recovered_result.loss_before == reference_result.loss_before
    assert recovered_result.loss_after == reference_result.loss_after
    assert all(status["version"] == 2 for status in recovered_statuses)
    assert all(status["checkpoint_versions"] == [0, 1, 2] for status in recovered_statuses)


def test_process_window_dead_worker_restarts_from_last_commit() -> None:
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

    reference_coordinator = _build_reference_coordinator(reference_shards)
    _ = asyncio.run(reference_coordinator.train_window(training_batch, version=0))
    reference_result = asyncio.run(reference_coordinator.train_window(training_batch, version=1))

    with ProcessWindowRunner(
        process_shards,
        learning_rate=1e-2,
        optimizer_name="adam",
        snapshot_depth=8,
    ) as runner:
        _ = runner.train_window(training_batch, version=0)
        original_pid = int(runner.process_status()[1]["pid"])
        runner._processes[1].terminate()
        runner._processes[1].wait(timeout=runner.request_timeout_s)

        with pytest.raises(RuntimeError, match="exited during request handling"):
            runner.train_window(training_batch, version=1)
        failed_statuses = runner.process_status()

        recovered_result = runner.train_window(training_batch, version=1)
        recovered_statuses = runner.process_status()

    restarted_pid = int(recovered_statuses[1]["pid"])
    assert int(failed_statuses[1]["pid"]) != original_pid
    assert restarted_pid != original_pid
    assert all(status["alive"] is True for status in failed_statuses)
    assert all(status["version"] == 1 for status in failed_statuses)
    assert all(status["alive"] is True for status in recovered_statuses)
    assert recovered_result.loss_before == reference_result.loss_before
    assert recovered_result.loss_after == reference_result.loss_after
    assert all(status["version"] == 2 for status in recovered_statuses)
