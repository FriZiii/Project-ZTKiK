"""Global statistical features in the spirit of NIST SP 800-22: bias, entropy, runs, byte histogram.

Each function maps packed samples to (values of shape (n_samples, k), k feature names).
"""

import numpy as np

from src.features.bits import blocks, shannon_entropy, to_bits, to_signs


def monobit_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NIST frequency (monobit) test: ones ratio and S_obs = |sum(+-1)| / sqrt(n)."""
    signs = to_signs(to_bits(samples))
    n = signs.shape[1]
    s_sum = signs.sum(axis=1, dtype=np.int64)
    return np.column_stack([(s_sum + n) / (2 * n), np.abs(s_sum) / np.sqrt(n)]), ["ones_ratio", "monobit_s_obs"]


def entropy_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Shannon entropy of non-overlapping 2-, 4- and 8-bit blocks, normalised to [0, 1]."""
    bits = to_bits(samples)
    columns, names = [], []
    for block_bits in (2, 4, 8):
        symbols = samples if block_bits == 8 else blocks(bits, block_bits)
        max_entropy = min(block_bits, np.log2(symbols.shape[1]))  # short sequences cannot reach block_bits
        columns.append(shannon_entropy(symbols.astype(np.int64), 1 << block_bits) / max_entropy)
        names.append(f"entropy_{block_bits}bit")
    return np.column_stack(columns), names


def runs_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """NIST runs test statistic and longest run of identical bits (normalised by log2 n)."""
    bits = to_bits(samples)
    n = bits.shape[1]
    changes = bits[:, 1:] != bits[:, :-1]
    runs = changes.sum(axis=1) + 1
    pi = bits.mean(axis=1)
    denom = 2 * np.sqrt(2 * n) * pi * (1 - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        runs_z = np.where(denom > 0, (runs - 2 * n * pi * (1 - pi)) / denom, 0.0)

    # Longest run: for every position, distance back to the start of the run it belongs to.
    idx = np.arange(1, n)
    run_start = np.maximum.accumulate(np.where(changes, idx, 0), axis=1)
    longest = (idx - run_start).max(axis=1) + 1
    return np.column_stack([runs_z, longest / np.log2(n)]), ["runs_z", "longest_run_norm"]


def byte_chi2_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Chi-square statistic of the byte histogram vs. uniform, divided by degrees of freedom."""
    n_rows, n_bytes = samples.shape
    offsets = np.arange(n_rows)[:, None] * 256
    counts = np.bincount((samples.astype(np.int64) + offsets).ravel(), minlength=n_rows * 256).reshape(n_rows, 256)
    expected = n_bytes / 256
    return (((counts - expected) ** 2 / expected).sum(axis=1) / 255)[:, None], ["byte_chi2_per_df"]
