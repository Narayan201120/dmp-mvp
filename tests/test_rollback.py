import numpy as np
import pytest

from training.checkpoints import SnapshotStore


def _state(value: float) -> dict[str, np.ndarray]:
    return {"weight": np.array([value], dtype=np.float32)}


def test_snapshot_store_restores_prior_versions() -> None:
    store = SnapshotStore(max_depth=4)
    store.save(0, _state(1.0))
    store.save(1, _state(1.5))
    store.save(2, _state(2.0))

    restored = store.restore(1)
    np.testing.assert_allclose(restored["weight"], np.array([1.5], dtype=np.float32))
    assert store.select_rollback_version(2) == 1


def test_snapshot_store_prunes_old_versions() -> None:
    store = SnapshotStore(max_depth=3)
    store.save(0, _state(0.0))
    store.save(1, _state(1.0))
    store.save(2, _state(2.0))
    store.save(3, _state(3.0))

    assert store.versions() == [1, 2, 3]
    with pytest.raises(KeyError):
        store.restore(0)


def test_snapshot_store_can_discard_future_versions() -> None:
    store = SnapshotStore(max_depth=4)
    store.save(0, _state(0.0))
    store.save(1, _state(1.0))
    store.save(2, _state(2.0))
    store.save(3, _state(3.0))

    store.discard_after(1)

    assert store.versions() == [0, 1]
    np.testing.assert_allclose(store.restore(1)["weight"], np.array([1.0], dtype=np.float32))
    with pytest.raises(KeyError):
        store.restore(2)
