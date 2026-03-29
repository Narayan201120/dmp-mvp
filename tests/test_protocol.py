import numpy as np
import pytest

from training.protocol import MissingBoundaryStateError, VersionMismatchError, open_window


def test_window_reconciles_adjacent_boundaries() -> None:
    session = open_window(version=4, shard_count=3, microbatches=2)

    left = session.materialize_boundary_state(0, 1, np.array([1.0, 2.0], dtype=np.float32))
    right = session.materialize_boundary_state(1, 2, np.array([3.0, 4.0], dtype=np.float32))

    session.exchange_boundary_state(left)
    session.exchange_boundary_state(right)
    reconciled = session.close_window()

    assert set(reconciled) == {(0, 1), (1, 2)}
    np.testing.assert_allclose(reconciled[(0, 1)].tensor, np.array([1.0, 2.0], dtype=np.float32))


def test_window_rejects_wrong_version() -> None:
    session = open_window(version=2, shard_count=3, microbatches=1)
    payload = session.materialize_boundary_state(0, 1, np.array([1.0], dtype=np.float32))
    wrong = payload.__class__(
        version=payload.version + 1,
        source_shard=payload.source_shard,
        target_shard=payload.target_shard,
        tensor=payload.tensor,
        checksum=payload.checksum,
    )

    with pytest.raises(VersionMismatchError):
        session.exchange_boundary_state(wrong)


def test_window_cannot_close_without_all_boundaries() -> None:
    session = open_window(version=7, shard_count=3, microbatches=1)
    session.exchange_boundary_state(
        session.materialize_boundary_state(0, 1, np.array([5.0], dtype=np.float32))
    )

    with pytest.raises(MissingBoundaryStateError):
        session.close_window()

