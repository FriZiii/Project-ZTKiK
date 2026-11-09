"""Shared helpers for feature extraction on packed bit sequences.

`samples` are uint8 arrays of shape (n_samples, n_bytes) in numpy.packbits big-endian layout.
All helpers are vectorised over samples (rows).
"""

import numpy as np


def to_bits(samples: np.ndarray) -> np.ndarray:
    return np.unpackbits(samples, axis=1)


def to_signs(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.int8) * 2 - 1  # 0/1 -> -1/+1


def autocorrelation(signs: np.ndarray, lag: int) -> np.ndarray:
    """Mean of s_i * s_{i+lag} for +-1 sequences: 0 for random data, +-1 for perfectly (anti)correlated."""
    if lag >= signs.shape[1]:
        return np.zeros(signs.shape[0])
    return (signs[:, :-lag].astype(np.int32) * signs[:, lag:]).mean(axis=1)


def shannon_entropy(symbols: np.ndarray, n_symbols: int) -> np.ndarray:
    """Per-row Shannon entropy in bits of integer symbols in [0, n_symbols)."""
    n_rows, length = symbols.shape
    offsets = np.arange(n_rows)[:, None] * n_symbols
    counts = np.bincount((symbols + offsets).ravel(), minlength=n_rows * n_symbols).reshape(n_rows, n_symbols)
    p = counts / length
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)


def blocks(bits: np.ndarray, block_bits: int) -> np.ndarray:
    """Non-overlapping block_bits-wide integers read from the bit sequence."""
    n_blocks = bits.shape[1] // block_bits
    weights = 1 << np.arange(block_bits - 1, -1, -1)
    return bits[:, : n_blocks * block_bits].reshape(bits.shape[0], n_blocks, block_bits) @ weights
