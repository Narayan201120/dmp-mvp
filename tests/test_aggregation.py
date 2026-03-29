import numpy as np

from training.aggregation import coordinate_median, trimmed_mean


def test_coordinate_median_resists_outlier() -> None:
    arrays = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.1, 0.9], dtype=np.float32),
        np.array([0.9, 1.05], dtype=np.float32),
        np.array([100.0, -100.0], dtype=np.float32),
    ]

    result = coordinate_median(arrays)
    np.testing.assert_allclose(result, np.array([1.05, 0.95], dtype=np.float32))


def test_trimmed_mean_discards_outlier() -> None:
    arrays = [
        np.array([1.0], dtype=np.float32),
        np.array([1.1], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([100.0], dtype=np.float32),
        np.array([1.05], dtype=np.float32),
    ]

    result = trimmed_mean(arrays, trim_ratio=0.2)
    np.testing.assert_allclose(result, np.array([1.05], dtype=np.float32), atol=1e-6)
