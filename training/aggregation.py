from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def coordinate_median(arrays: Sequence[np.ndarray]) -> np.ndarray:
    stack = _stack(arrays)
    return np.median(stack, axis=0).astype(np.float32)


def trimmed_mean(arrays: Sequence[np.ndarray], trim_ratio: float = 0.2) -> np.ndarray:
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0.0, 0.5)")
    stack = np.sort(_stack(arrays), axis=0)
    trim = int(np.floor(stack.shape[0] * trim_ratio))
    if trim * 2 >= stack.shape[0]:
        raise ValueError("trim_ratio removes every sample")
    if trim:
        stack = stack[trim : stack.shape[0] - trim]
    return np.mean(stack, axis=0).astype(np.float32)


def _stack(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one array is required")
    coerced = [np.asarray(array, dtype=np.float32) for array in arrays]
    return np.stack(coerced, axis=0)

