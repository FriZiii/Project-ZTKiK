"""Univariate separability of every feature: how well it alone distinguishes a weak generator from CSPRNG."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.prng.weak.lcg import DEFAULT_OUTPUT_SHIFT

REFERENCE = "csprng"
MONOBIT_FEATURE = "monobit_s_obs"  # NIST SP 800-22 frequency test statistic, the classical baseline


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic, ties get average ranks."""
    values = np.concatenate([positive, negative]).astype(np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def distinguishing_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """max(AUC, 1 - AUC): a distinguisher may use either tail of the feature, so 0.5 is chance and 1 is perfect."""
    auc = roc_auc(positive, negative)
    return max(auc, 1.0 - auc)


def output_shift(generator: str) -> int:
    match = re.search(r"_shift(\d+)", generator)
    return int(match.group(1)) if match else DEFAULT_OUTPUT_SHIFT


def weak_generators(features_root: Path | str) -> list[str]:
    """Weak generators with extracted features, most defective (smallest output shift) first."""
    names = [path.name for path in Path(features_root).iterdir() if path.is_dir() and path.name != REFERENCE]
    return sorted(names, key=output_shift)


def display_name(generator: str) -> str:
    return f"LCG, przesunięcie {output_shift(generator)}"


def load_features(features_root: Path | str, generator: str, n_bits: int):
    with np.load(Path(features_root) / generator / f"n{n_bits}.npz") as data:
        return data["X"], list(data["feature_names"]), list(data["feature_groups"])


def separability_table(features_root: Path | str) -> pd.DataFrame:
    """One row per (weak generator, length, feature) with its distinguishing AUC against CSPRNG."""
    rows = []
    for generator in weak_generators(features_root):
        lengths = sorted(int(path.stem[1:]) for path in (Path(features_root) / generator).glob("n*.npz"))
        for n_bits in lengths:
            weak, names, groups = load_features(features_root, generator, n_bits)
            secure, secure_names, _ = load_features(features_root, REFERENCE, n_bits)
            if names != secure_names:
                raise ValueError(f"feature mismatch for n={n_bits}: re-run `main.py features` for all generators")
            for j, (name, group) in enumerate(zip(names, groups)):
                rows.append({"generator": generator, "n_bits": n_bits, "group": group, "feature": name,
                             "auc": distinguishing_auc(weak[:, j], secure[:, j])})
    return pd.DataFrame(rows)
