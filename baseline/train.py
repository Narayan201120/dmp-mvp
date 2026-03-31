from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from training.reference import (
    build_char_vocab,
    encode_text,
    iter_eval_windows,
    load_eval_artifact,
    sample_batch,
    write_curve,
    write_json,
)

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
    write_json(Path(config.output_summary), summary)
    return summary


def main() -> None:
    config = parse_args()
    summary = run(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
