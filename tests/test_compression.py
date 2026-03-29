import numpy as np

from training.compression import (
    compress_boundary_payload,
    cosine_similarity,
    decompress_boundary_payload,
)


def test_compression_round_trip_preserves_signal() -> None:
    payload = np.array([0.0, 0.0, 4.0, -3.0, 0.0, 1.0], dtype=np.float32)
    compressed, error = compress_boundary_payload(payload, topk_ratio=0.5, num_bits=8)
    restored = decompress_boundary_payload(compressed)

    assert cosine_similarity(payload, restored) > 0.95
    assert error.shape == payload.shape


def test_error_feedback_recovers_deferred_signal() -> None:
    payload = np.array([1.0, 0.1, -0.4, 0.0], dtype=np.float32)

    first, first_error = compress_boundary_payload(payload, topk_ratio=0.25, num_bits=8)
    second, second_error = compress_boundary_payload(
        np.zeros_like(payload), topk_ratio=0.25, num_bits=8, error_feedback=first_error
    )

    restored_once = decompress_boundary_payload(first)
    restored_twice = restored_once + decompress_boundary_payload(second)

    assert np.linalg.norm(payload - restored_twice) < np.linalg.norm(payload - restored_once)
    assert np.linalg.norm(second_error) <= np.linalg.norm(first_error)

