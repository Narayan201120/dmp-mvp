from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import sys

import torch
from torch import optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.metrics import next_token_loss, split_next_token_batch
from training.model_factory import ToyTransformerConfig, build_toy_transformer

EVAL_CONFIG_PATH = PROJECT_ROOT / "eval" / "config.json"
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.txt"
EVAL_HASH_PATH = PROJECT_ROOT / "eval" / "eval_hash.txt"
DEFAULT_OUTPUT_CURVE = PROJECT_ROOT / "baseline" / "baseline_loss_curve.csv"
DEFAULT_OUTPUT_SUMMARY = PROJECT_ROOT / "baseline" / "baseline_config.json"


@dataclass(slots=True)
class BaselineConfig:
    output_curve: str = str(DEFAULT_OUTPUT_CURVE)
    output_summary: str = str(DEFAULT_OUTPUT_SUMMARY)
    seed: int = 7
    steps: int = 30
    eval_every: int = 5
    learning_rate: float = 3e-3
    d_model: int = 32
    num_heads: int = 4
    mlp_hidden_dim: int = 64
    num_layers: int = 6
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class EvalArtifact:
    text: str
    seq_len: int
    batch_size: int


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Run the centralized Phase 0 baseline.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-curve", default=str(DEFAULT_OUTPUT_CURVE))
    parser.add_argument("--output-summary", default=str(DEFAULT_OUTPUT_SUMMARY))
    args = parser.parse_args()
    return BaselineConfig(
        output_curve=args.output_curve,
        output_summary=args.output_summary,
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )


def load_eval_artifact() -> EvalArtifact:
    config = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "locked":
        raise RuntimeError("eval artifact must be locked before running the baseline")
    if config.get("tokenizer") != "character":
        raise RuntimeError("baseline/train.py currently supports character tokenization only")

    text = EVAL_SET_PATH.read_text(encoding="utf-8").rstrip("\n")
    expected_hash = EVAL_HASH_PATH.read_text(encoding="utf-8").strip()
    actual_hash = compute_sha256(EVAL_SET_PATH)
    if expected_hash != actual_hash:
        raise RuntimeError("eval_set.txt hash does not match eval_hash.txt")
    if len(text) < 3:
        raise RuntimeError("eval text is too short for next-token training")

    seq_len = int(config["seq_len"])
    batch_size = int(config["batch_size"])
    if seq_len < 2:
        raise RuntimeError("eval seq_len must be at least 2")
    if batch_size < 1:
        raise RuntimeError("eval batch_size must be positive")

    return EvalArtifact(text=text, seq_len=seq_len, batch_size=batch_size)


def compute_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    alphabet = sorted(set(text))
    stoi = {char: index for index, char in enumerate(alphabet)}
    itos = {index: char for char, index in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[char] for char in text], dtype=torch.long)


def sample_batch(
    encoded: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    max_start = encoded.size(0) - (sequence_length + 1)
    if max_start < 0:
        raise RuntimeError("encoded text is too short for the requested sequence length")

    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    windows = [encoded[start : start + sequence_length + 1] for start in starts]
    return torch.stack(windows, dim=0).to(device)


def iter_eval_windows(encoded: torch.Tensor, *, sequence_length: int, device: torch.device) -> list[torch.Tensor]:
    windows: list[torch.Tensor] = []
    max_start = encoded.size(0) - (sequence_length + 1)
    if max_start < 0:
        raise RuntimeError("encoded text is too short for evaluation")

    step = max(1, sequence_length)
    for start in range(0, max_start + 1, step):
        windows.append(encoded[start : start + sequence_length + 1].unsqueeze(0).to(device))

    if max_start % step != 0:
        windows.append(encoded[max_start : max_start + sequence_length + 1].unsqueeze(0).to(device))

    return windows


def evaluate(model: torch.nn.Module, eval_windows: list[torch.Tensor]) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for window in eval_windows:
            inputs, targets = split_next_token_batch(window)
            logits = model(inputs)
            losses.append(float(next_token_loss(logits, targets).item()))
    model.train()
    return sum(losses) / len(losses)


def write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "eval_loss"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(config: BaselineConfig) -> dict[str, object]:
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    artifact = load_eval_artifact()
    stoi, _ = build_char_vocab(artifact.text)
    encoded = encode_text(artifact.text, stoi)
    device = torch.device(config.device)
    sequence_length = min(artifact.seq_len, encoded.size(0) - 1)
    eval_windows = iter_eval_windows(encoded, sequence_length=sequence_length, device=device)

    model = build_toy_transformer(
        ToyTransformerConfig(
            vocab_size=len(stoi),
            max_seq_len=sequence_length,
            d_model=config.d_model,
            num_heads=config.num_heads,
            mlp_hidden_dim=config.mlp_hidden_dim,
            num_layers=config.num_layers,
        ),
        seed=config.seed,
        device=device,
    )
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    rows: list[dict[str, float]] = []
    initial_eval_loss = evaluate(model, eval_windows)
    rows.append({"step": 0, "train_loss": initial_eval_loss, "eval_loss": initial_eval_loss})

    train_loss = initial_eval_loss
    for step in range(1, config.steps + 1):
        batch = sample_batch(
            encoded,
            batch_size=artifact.batch_size,
            sequence_length=sequence_length,
            rng=rng,
            device=device,
        )
        inputs, targets = split_next_token_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = next_token_loss(logits, targets)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

        if step % config.eval_every == 0 or step == config.steps:
            eval_loss = evaluate(model, eval_windows)
            rows.append({"step": step, "train_loss": train_loss, "eval_loss": eval_loss})

    output_path = Path(config.output_curve)
    write_curve(output_path, rows)
    final_eval_loss = float(rows[-1]["eval_loss"])

    summary = {
        "config": asdict(config),
        "dataset": "shakespeare_public_domain_excerpt_v1",
        "vocab_size": len(stoi),
        "sequence_length": sequence_length,
        "batch_size": artifact.batch_size,
        "initial_eval_loss": float(initial_eval_loss),
        "final_train_loss": float(train_loss),
        "final_eval_loss": final_eval_loss,
        "curve_path": str(output_path),
    }
    write_summary(Path(config.output_summary), summary)
    return summary


def main() -> None:
    config = parse_args()
    summary = run(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
