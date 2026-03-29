from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json


@dataclass(slots=True)
class BaselineConfig:
    output_curve: str = "baseline/baseline_loss_curve.csv"
    seed: int = 0
    steps: int = 0


def write_placeholder_curve(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "loss"])


def main() -> None:
    config = BaselineConfig()
    write_placeholder_curve(Path(config.output_curve))
    print(json.dumps(asdict(config), indent=2))
    raise SystemExit(
        "Baseline scaffold created. Implement centralized training before using this script."
    )


if __name__ == "__main__":
    main()

