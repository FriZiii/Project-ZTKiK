"""Feature registry: each group maps packed samples to (values (n_samples, k), k feature names)."""

import numpy as np

from src.features.statistical import byte_chi2_features, entropy_features, monobit_features, runs_features

FEATURE_GROUPS = {
    "monobit": monobit_features,
    "entropy": entropy_features,
    "runs": runs_features,
    "byte_chi2": byte_chi2_features,
}


def extract_features(samples: np.ndarray, groups: list[str] | None = None,
                     batch_size: int = 5000) -> tuple[np.ndarray, list[str], list[str]]:
    """Concatenate the selected feature groups (all by default) into one float32 matrix.

    Returns (X, feature names, group name of every feature).
    """
    groups = list(FEATURE_GROUPS) if groups is None else groups
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")

    batches, names, name_groups = [], [], []
    for start in range(0, samples.shape[0], batch_size):  # bounded memory for long sequences
        batch = samples[start:start + batch_size]
        parts = [FEATURE_GROUPS[group](batch) for group in groups]
        batches.append(np.hstack([values for values, _ in parts]).astype(np.float32))
        names = [name for _, group_names in parts for name in group_names]
        name_groups = [group for group, (_, group_names) in zip(groups, parts) for _ in group_names]
    return np.vstack(batches), names, name_groups
