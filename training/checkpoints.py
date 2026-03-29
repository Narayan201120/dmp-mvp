from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping

import numpy as np


StateDict = Mapping[str, np.ndarray]


def clone_state(state: StateDict) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=np.float32).copy() for key, value in state.items()}


def subtract_state(lhs: StateDict, rhs: StateDict) -> dict[str, np.ndarray]:
    _ensure_same_keys(lhs, rhs)
    return {key: np.asarray(lhs[key], dtype=np.float32) - np.asarray(rhs[key], dtype=np.float32) for key in lhs}


def add_delta(state: StateDict, delta: StateDict) -> dict[str, np.ndarray]:
    _ensure_same_keys(state, delta)
    return {key: np.asarray(state[key], dtype=np.float32) + np.asarray(delta[key], dtype=np.float32) for key in state}


class SnapshotStore:
    """Stores a rolling window of state deltas and reconstructs snapshots on demand."""

    def __init__(self, max_depth: int = 10) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.max_depth = max_depth
        self._base_version: int | None = None
        self._base_state: dict[str, np.ndarray] | None = None
        self._head_state: dict[str, np.ndarray] | None = None
        self._deltas: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def save(self, version: int, state: StateDict) -> None:
        if version < 0:
            raise ValueError("version must be non-negative")
        current_state = clone_state(state)
        if self._base_state is None:
            self._base_version = version
            self._base_state = current_state
            self._head_state = current_state
            return
        if version <= self.latest_version():
            raise ValueError("version must increase monotonically")
        assert self._head_state is not None
        self._deltas[version] = subtract_state(current_state, self._head_state)
        self._head_state = current_state
        self._prune()

    def restore(self, version: int) -> dict[str, np.ndarray]:
        if self._base_version is None or self._base_state is None:
            raise KeyError("no snapshots have been saved")
        if version < self._base_version or version > self.latest_version():
            raise KeyError(f"version {version} is outside the retained snapshot window")
        restored = clone_state(self._base_state)
        for delta_version, delta in self._deltas.items():
            if delta_version > version:
                break
            restored = add_delta(restored, delta)
        return restored

    def latest_version(self) -> int:
        if self._base_version is None:
            raise KeyError("no snapshots have been saved")
        if not self._deltas:
            return self._base_version
        return next(reversed(self._deltas))

    def versions(self) -> list[int]:
        if self._base_version is None:
            return []
        return [self._base_version, *self._deltas.keys()]

    def get_snapshot_depth(self) -> int:
        return len(self.versions())

    def select_rollback_version(self, failing_version: int) -> int:
        candidates = [version for version in self.versions() if version < failing_version]
        if not candidates:
            raise KeyError("no rollback target exists before the failing version")
        return candidates[-1]

    def _prune(self) -> None:
        assert self._base_version is not None
        assert self._base_state is not None
        while self.get_snapshot_depth() > self.max_depth:
            oldest_version, oldest_delta = self._deltas.popitem(last=False)
            self._base_state = add_delta(self._base_state, oldest_delta)
            self._base_version = oldest_version


def _ensure_same_keys(lhs: Mapping[str, np.ndarray], rhs: Mapping[str, np.ndarray]) -> None:
    if set(lhs) != set(rhs):
        raise ValueError("state dictionaries must contain the same keys")

