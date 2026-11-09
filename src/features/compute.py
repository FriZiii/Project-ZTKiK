import time
from pathlib import Path

import numpy as np

from src.dataset.storage import DatasetStorage
from src.features.extract import extract_features


def available_datasets(raw_root: Path | str) -> list[tuple[str, int]]:
    """(generator, n_bits) pairs present under <raw_root>/<generator>/n<bits>.npz."""
    return sorted(
        (path.parent.name, int(path.stem[1:]))
        for path in Path(raw_root).glob("*/n*.npz")
    )


def compute_features(raw_root: str, output: str, generators: list[str] | None = None,
                     lengths: list[int] | None = None, groups: list[str] | None = None) -> None:
    """Layout: <output>/<generator>/n<bits>.npz with X (float32), feature_names, feature_groups, label, seeds."""
    storage = DatasetStorage(raw_root)
    for generator, n_bits in available_datasets(raw_root):
        if (generators and generator not in generators) or (lengths and n_bits not in lengths):
            continue
        start = time.perf_counter()
        dataset = storage.load(generator, n_bits)
        features, names, name_groups = extract_features(dataset.samples, groups)

        path = Path(output) / generator / f"n{n_bits}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=features, feature_names=np.array(names),
                            feature_groups=np.array(name_groups), label=dataset.label,
                            seeds=dataset.seeds)
        print(f"{generator} | n={n_bits:>5} bits | {features.shape[1]} features | "
              f"{time.perf_counter() - start:.2f}s -> {path}")
