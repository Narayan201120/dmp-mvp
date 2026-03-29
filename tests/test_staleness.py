import math

import pytest

from training.staleness import decay_weight, should_drop, version_distance


def test_staleness_drop_threshold() -> None:
    assert not should_drop(message_version=8, current_version=10, max_staleness=2)
    assert should_drop(message_version=7, current_version=10, max_staleness=2)


def test_decay_weight_is_monotonic() -> None:
    current = 12
    fresh = decay_weight(12, current, max_staleness=4, decay_rate=0.5)
    stale = decay_weight(10, current, max_staleness=4, decay_rate=0.5)

    assert math.isclose(fresh, 1.0)
    assert 0.0 < stale < fresh


def test_future_version_is_rejected() -> None:
    with pytest.raises(ValueError):
        version_distance(current_version=3, message_version=4)

