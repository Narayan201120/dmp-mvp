from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CompressedBoundaryPayload:
    shape: tuple[int, ...]
    indices: np.ndarray
    quantized_values: np.ndarray
    scale: float
    num_bits: int


def topk_sparsify(array: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0.0, 1.0]")

    flat = np.asarray(array, dtype=np.float32).ravel()
    keep = max(1, int(np.ceil(flat.size * ratio)))

    if keep == flat.size:
        indices = np.arange(flat.size, dtype=np.int64)
    else:
        indices = np.argpartition(np.abs(flat), -keep)[-keep:]
        indices = np.sort(indices.astype(np.int64))

    values = flat[indices].astype(np.float32)
    return indices, values


def quantize(values: np.ndarray, num_bits: int) -> tuple[np.ndarray, float]:
    if not 1 <= num_bits <= 16:
        raise ValueError("num_bits must be between 1 and 16")

    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=np.int16), 1.0

    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return np.zeros(values.shape, dtype=np.int16), 1.0

    qmax = max((1 << (num_bits - 1)) - 1, 1)
    scale = max_abs / qmax
    quantized = np.clip(np.round(values / scale), -qmax, qmax).astype(np.int16)
    return quantized, float(scale)


def dequantize(values: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(scale)


def compress_boundary_payload(
    array: np.ndarray,
    *,
    topk_ratio: float = 1.0,
    num_bits: int = 8,
    error_feedback: np.ndarray | None = None,
) -> tuple[CompressedBoundaryPayload, np.ndarray]:
    residual = np.asarray(array, dtype=np.float32).copy()
    if error_feedback is not None:
        residual += np.asarray(error_feedback, dtype=np.float32)

    indices, values = topk_sparsify(residual, topk_ratio)
    quantized_values, scale = quantize(values, num_bits)
    payload = CompressedBoundaryPayload(
        shape=residual.shape,
        indices=indices,
        quantized_values=quantized_values,
        scale=scale,
        num_bits=num_bits,
    )

    reconstructed = decompress_boundary_payload(payload)
    next_error = residual - reconstructed
    return payload, next_error


def decompress_boundary_payload(payload: CompressedBoundaryPayload) -> np.ndarray:
    flat = np.zeros(int(np.prod(payload.shape)), dtype=np.float32)
    flat[payload.indices] = dequantize(payload.quantized_values, payload.scale)
    return flat.reshape(payload.shape)


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_vec = np.asarray(lhs, dtype=np.float32).ravel()
    rhs_vec = np.asarray(rhs, dtype=np.float32).ravel()
    lhs_norm = float(np.linalg.norm(lhs_vec))
    rhs_norm = float(np.linalg.norm(rhs_vec))
    if lhs_norm == 0.0 or rhs_norm == 0.0:
        return 1.0 if lhs_norm == rhs_norm else 0.0
    return float(np.dot(lhs_vec, rhs_vec) / (lhs_norm * rhs_norm))

