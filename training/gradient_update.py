from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(slots=True)
class GradientUpdate:
    shard_id: int
    base_version: int
    gradients: Mapping[str, Any]
    grad_norm: float
    loss_before: float
    loss_after: float
    data_hash: str
    timestamp: float | None = None
    node_id: str | None = None

    def is_valid(
        self,
        *,
        max_norm: float = 10.0,
        max_staleness: int = 50,
        current_version: int = 0,
    ) -> bool:
        staleness = current_version - self.base_version
        return (
            self.grad_norm >= 0.0
            and self.grad_norm < max_norm
            and self.loss_after < self.loss_before
            and 0 <= staleness <= max_staleness
        )

