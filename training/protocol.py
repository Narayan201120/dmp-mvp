from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, Tuple

import numpy as np
import torch


class ProtocolError(RuntimeError):
    """Raised when a window protocol invariant is violated."""


class VersionMismatchError(ProtocolError):
    """Raised when a boundary payload does not match the active window version."""


class MissingBoundaryStateError(ProtocolError):
    """Raised when a window is closed before all required boundaries are exchanged."""


@dataclass(frozen=True, slots=True)
class WindowSpec:
    version: int
    shard_count: int
    microbatches: int

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("version must be non-negative")
        if self.shard_count < 2:
            raise ValueError("shard_count must be at least 2")
        if self.microbatches < 1:
            raise ValueError("microbatches must be at least 1")


@dataclass(frozen=True, slots=True)
class BoundaryPayload:
    version: int
    source_shard: int
    target_shard: int
    tensor: np.ndarray
    checksum: str


@dataclass(slots=True)
class WindowSession:
    spec: WindowSpec
    _payloads: Dict[Tuple[int, int], BoundaryPayload] = field(default_factory=dict)
    _closed: bool = False

    @property
    def expected_edges(self) -> set[tuple[int, int]]:
        return {(shard_id, shard_id + 1) for shard_id in range(self.spec.shard_count - 1)}

    def materialize_boundary_state(
        self, source_shard: int, target_shard: int, tensor: np.ndarray
    ) -> BoundaryPayload:
        _validate_edge(source_shard, target_shard, self.spec.shard_count)
        array = np.asarray(tensor, dtype=np.float32).copy()
        return BoundaryPayload(
            version=self.spec.version,
            source_shard=source_shard,
            target_shard=target_shard,
            tensor=array,
            checksum=_checksum(array),
        )

    def exchange_boundary_state(self, payload: BoundaryPayload) -> None:
        if self._closed:
            raise ProtocolError("window is already closed")
        if payload.version != self.spec.version:
            raise VersionMismatchError(
                f"payload version {payload.version} does not match active version {self.spec.version}"
            )
        _validate_edge(payload.source_shard, payload.target_shard, self.spec.shard_count)
        if payload.checksum != _checksum(payload.tensor):
            raise ProtocolError("boundary payload checksum mismatch")
        self._payloads[(payload.source_shard, payload.target_shard)] = payload

    def reconcile_window(self) -> dict[tuple[int, int], BoundaryPayload]:
        missing = self.expected_edges.difference(self._payloads)
        if missing:
            missing_edges = ", ".join(f"{src}->{dst}" for src, dst in sorted(missing))
            raise MissingBoundaryStateError(f"missing boundary state for edges: {missing_edges}")
        return dict(sorted(self._payloads.items()))

    def close_window(self) -> dict[tuple[int, int], BoundaryPayload]:
        reconciled = self.reconcile_window()
        self._closed = True
        return reconciled


def open_window(version: int, shard_count: int, microbatches: int) -> WindowSession:
    return WindowSession(spec=WindowSpec(version=version, shard_count=shard_count, microbatches=microbatches))


def materialize_tensor_boundary_state(
    session: WindowSession,
    source_shard: int,
    target_shard: int,
    tensor: torch.Tensor,
) -> BoundaryPayload:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return session.materialize_boundary_state(source_shard, target_shard, array)


def payload_as_tensor(
    payload: BoundaryPayload,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(payload.tensor.copy()).to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _validate_edge(source_shard: int, target_shard: int, shard_count: int) -> None:
    if source_shard < 0 or target_shard < 0:
        raise ValueError("shard ids must be non-negative")
    if source_shard >= shard_count or target_shard >= shard_count:
        raise ValueError("shard id exceeds configured shard count")
    if target_shard != source_shard + 1:
        raise ValueError("boundary exchange must occur between adjacent shards only")


def _checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
