from __future__ import annotations

import math


def version_distance(current_version: int, message_version: int) -> int:
    if current_version < message_version:
        raise ValueError("message version cannot be ahead of the current version")
    return current_version - message_version


def should_drop(message_version: int, current_version: int, max_staleness: int) -> bool:
    if max_staleness < 0:
        raise ValueError("max_staleness must be non-negative")
    return version_distance(current_version, message_version) > max_staleness


def decay_weight(
    message_version: int,
    current_version: int,
    *,
    max_staleness: int,
    decay_rate: float,
    floor: float = 0.0,
) -> float:
    if decay_rate < 0.0:
        raise ValueError("decay_rate must be non-negative")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0.0, 1.0]")
    if should_drop(message_version, current_version, max_staleness):
        return 0.0
    distance = version_distance(current_version, message_version)
    return max(floor, math.exp(-decay_rate * distance))

