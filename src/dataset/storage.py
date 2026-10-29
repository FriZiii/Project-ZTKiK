import json
from pathlib import Path

import numpy as np

from src.dataset.builder import GeneratedDataset


class DatasetStorage:
    """Layout: <root>/<generator>/n<bits>.npz (arrays) + n<bits>.json (metadata)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _base_path(self, generator_name: str, n_bits: int) -> Path:
        return self.root / generator_name / f"n{n_bits}"

    def save(self, dataset: GeneratedDataset) -> Path:
        base = self._base_path(dataset.generator_name, dataset.n_bits)
        base.parent.mkdir(parents=True, exist_ok=True)

        npz_path = base.with_suffix(".npz")
        np.savez_compressed(npz_path, samples=dataset.samples, seeds=dataset.seeds)

        metadata = {
            "generator": dataset.generator_name,
            "generator_params": dataset.generator_params,
            "label": dataset.label,
            "n_bits": dataset.n_bits,
            "n_samples": dataset.n_samples,
            "bit_packing": "numpy.packbits, big-endian",
            "created_at": dataset.created_at,
        }
        base.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return npz_path

    def load(self, generator_name: str, n_bits: int) -> GeneratedDataset:
        base = self._base_path(generator_name, n_bits)
        metadata = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(base.with_suffix(".npz")) as arrays:
            return GeneratedDataset(
                generator_name=metadata["generator"],
                generator_params=metadata["generator_params"],
                label=metadata["label"],
                n_bits=metadata["n_bits"],
                samples=arrays["samples"],
                seeds=arrays["seeds"],
                created_at=metadata["created_at"],
            )
