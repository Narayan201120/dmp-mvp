from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class NodeState:
    node_id: str
    shard_id: int
    version: int = 0
    latest_eval_loss: float | None = None
    last_checkpoint_version: int | None = None

