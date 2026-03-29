from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    base_latency_ms: int = 0
    jitter_ms: int = 0
    packet_loss: float = 0.0
    reorder_chance: float = 0.0

    def __post_init__(self) -> None:
        if self.base_latency_ms < 0 or self.jitter_ms < 0:
            raise ValueError("latency values must be non-negative")
        if not 0.0 <= self.packet_loss <= 1.0:
            raise ValueError("packet_loss must be in [0.0, 1.0]")
        if not 0.0 <= self.reorder_chance <= 1.0:
            raise ValueError("reorder_chance must be in [0.0, 1.0]")


def sample_delivery_delay(config: NetworkConfig, rng: random.Random | None = None) -> int | None:
    generator = rng or random.Random()
    if generator.random() < config.packet_loss:
        return None
    jitter = generator.randint(-config.jitter_ms, config.jitter_ms) if config.jitter_ms else 0
    delay = max(0, config.base_latency_ms + jitter)
    if config.reorder_chance and generator.random() < config.reorder_chance:
        delay += max(1, config.base_latency_ms + config.jitter_ms + 1)
    return delay


def wrap_message(payload: Any, *, delay_ms: int | None) -> dict[str, Any]:
    return {"payload": payload, "delay_ms": delay_ms}
