import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from experiments.distributed_train import DistributedRunConfig, run
from training.reference import read_curve_rows, write_curve, write_json


def _write_baseline_reference(
    root: Path,
    *,
    steps: int,
    eval_every: int,
    seed: int,
    learning_rate: float,
    d_model: int,
    num_heads: int,
    mlp_hidden_dim: int,
    num_layers: int,
    device: str,
    final_eval_loss: float,
) -> tuple[Path, Path]:
    curve_path = root / "baseline_curve.csv"
    summary_path = root / "baseline_summary.json"
    write_curve(
        curve_path,
        [
            {"step": 0, "train_loss": final_eval_loss + 1.0, "eval_loss": final_eval_loss + 1.0},
            {"step": steps, "train_loss": final_eval_loss + 0.1, "eval_loss": final_eval_loss},
        ],
    )
    write_json(
        summary_path,
        {
            "config": {
                "steps": steps,
                "eval_every": eval_every,
                "seed": seed,
                "learning_rate": learning_rate,
                "d_model": d_model,
                "num_heads": num_heads,
                "mlp_hidden_dim": mlp_hidden_dim,
                "num_layers": num_layers,
                "device": device,
            },
            "dataset": "shakespeare_public_domain_excerpt_v1",
            "final_eval_loss": final_eval_loss,
        },
    )
    return curve_path, summary_path


def _make_workspace_tmp_dir() -> Path:
    root = Path.cwd() / "tests_runtime_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    return path


def test_distributed_runner_writes_baseline_compatible_artifacts() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        baseline_curve, baseline_summary = _write_baseline_reference(
            tmp_dir,
            steps=2,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            final_eval_loss=2.25,
        )
        output_curve = tmp_dir / "distributed_curve.csv"
        output_summary = tmp_dir / "distributed_summary.json"
        config = DistributedRunConfig(
            output_curve=str(output_curve),
            output_summary=str(output_summary),
            baseline_curve=str(baseline_curve),
            baseline_summary=str(baseline_summary),
            steps=2,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            num_shards=2,
            snapshot_depth=8,
        )

        summary = run(config)

        rows = read_curve_rows(output_curve)
        assert [row["step"] for row in rows] == [0.0, 1.0, 2.0]
        assert summary["curve_path"] == str(output_curve)
        assert summary["final_version"] == 2
        assert summary["boundary_events_recorded"] == 2
        assert summary["boundary_delivery"]["events_recorded"] == 2
        assert summary["boundary_delivery"]["status_counts"] == {"delivered": 2}
        assert summary["boundary_delivery"]["delay_ms"] == {"min": 0, "max": 0, "avg": 0.0}
        assert summary["boundary_delivery"]["max_simulated_current_version"] == 1
        assert summary["boundary_delivery"]["staleness_multiplier"] == {"min": 1.0, "max": 1.0, "avg": 1.0}
        assert summary["boundary_delivery"]["compression"] == {
            "applied_events": 0,
            "min_values": 8192,
            "max_values": 8192,
            "avg_values": 8192.0,
            "wire_bytes": {"min": 32768, "max": 32768, "avg": 32768.0},
            "dense_wire_bytes": {"min": 32768, "max": 32768, "avg": 32768.0},
            "wire_ratio": {"min": 1.0, "max": 1.0, "avg": 1.0},
        }
        assert summary["baseline_reference"]["config_matches"] is True
        assert summary["baseline_reference"]["schedule_matches"] is True

        written_summary = json.loads(output_summary.read_text(encoding="utf-8"))
        assert written_summary["final_eval_loss"] == summary["final_eval_loss"]
        assert written_summary["baseline_reference"]["final_eval_loss"] == 2.25
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_runner_rejects_mismatched_frozen_baseline() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        baseline_curve, baseline_summary = _write_baseline_reference(
            tmp_dir,
            steps=3,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            final_eval_loss=2.25,
        )
        config = DistributedRunConfig(
            output_curve=str(tmp_dir / "distributed_curve.csv"),
            output_summary=str(tmp_dir / "distributed_summary.json"),
            baseline_curve=str(baseline_curve),
            baseline_summary=str(baseline_summary),
            steps=2,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            num_shards=2,
            snapshot_depth=8,
        )

        with pytest.raises(ValueError, match="does not match the frozen baseline"):
            run(config)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_distributed_runner_rejects_boundary_activation_error_feedback() -> None:
    tmp_dir = _make_workspace_tmp_dir()
    try:
        baseline_curve, baseline_summary = _write_baseline_reference(
            tmp_dir,
            steps=2,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            final_eval_loss=2.25,
        )
        config = DistributedRunConfig(
            output_curve=str(tmp_dir / "distributed_curve.csv"),
            output_summary=str(tmp_dir / "distributed_summary.json"),
            baseline_curve=str(baseline_curve),
            baseline_summary=str(baseline_summary),
            steps=2,
            eval_every=1,
            seed=7,
            learning_rate=3e-3,
            d_model=16,
            num_heads=4,
            mlp_hidden_dim=32,
            num_layers=4,
            device="cpu",
            num_shards=2,
            snapshot_depth=8,
            compression_topk_ratio=0.25,
            compression_num_bits=8,
            compression_error_feedback=True,
        )

        with pytest.raises(ValueError, match="not supported for boundary activations"):
            run(config)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
