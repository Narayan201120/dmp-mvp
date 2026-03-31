from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_CONFIG_PATH = PROJECT_ROOT / "eval" / "config.json"
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "eval_set.txt"
EVAL_HASH_PATH = PROJECT_ROOT / "eval" / "eval_hash.txt"


@dataclass(frozen=True, slots=True)
class EvalArtifact:
    text: str
    seq_len: int
    batch_size: int


def load_eval_artifact() -> EvalArtifact:
    config = json.loads(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "locked":
        raise RuntimeError("eval artifact must be locked before training")
    if config.get("tokenizer") != "character":
        raise RuntimeError("Phase 0 training currently supports character tokenization only")

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


def write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "eval_loss"])
        writer.writeheader()
        writer.writerows(rows)


def read_curve_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "step": float(row["step"]),
                "train_loss": float(row["train_loss"]),
                "eval_loss": float(row["eval_loss"]),
            }
            for row in reader
        ]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
