"""Correlation features: dependencies between bits at fixed distances.

Under Kerckhoffs' principle the attacker knows the generator emits bytes, so the bit-plane features
look at the 8 byte-aligned bit positions separately (plane 0 = least significant bit).
Each function maps packed samples to (values of shape (n_samples, k), k feature names).
"""

import numpy as np

from src.features.bits import autocorrelation, to_bits, to_signs

# Lags in bits (whole stream) and in bytes (bit planes); lags not shorter than the sequence give 0.
BIT_LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32)
PLANE_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def autocorrelation_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Autocorrelation of the whole bit stream at short lags (bit-level dependencies)."""
    signs = to_signs(to_bits(samples))
    return (np.column_stack([autocorrelation(signs, lag) for lag in BIT_LAGS]),
            [f"autocorr_bit_lag{lag}" for lag in BIT_LAGS])


def bit_plane_features(samples: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """For each of the 8 bit positions within a byte: bias and autocorrelation across bytes.

    Targets short-period low bits of power-of-two LCGs (state bit j has period 2**(j+1)):
    e.g. the LSB of a raw LCG byte alternates, giving autocorrelation -1 at lag 1.
    Lags are powers of two because that is where those periodicities show up.
    """
    columns, names = [], []
    for plane in range(8):
        signs = to_signs((samples >> plane) & 1)
        columns.append(signs.mean(axis=1))
        names.append(f"plane{plane}_bias")
        for lag in PLANE_LAGS:
            columns.append(autocorrelation(signs, lag))
            names.append(f"plane{plane}_autocorr_lag{lag}")
    return np.column_stack(columns), names
