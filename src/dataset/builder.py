from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from src.prng.base import BaseGenerator

NO_SEED = -1


@dataclass
class GeneratedDataset:
    generator_name: str
    generator_params: dict
    label: int
    n_bits: int
    samples: np.ndarray  # uint8, shape (n_samples, n_bits // 8), packed bits
    seeds: np.ndarray  # int64, shape (n_samples,)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    def unpacked_bits(self) -> np.ndarray:
        return np.unpackbits(self.samples, axis=1)


class DatasetBuilder:
    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    def build(self, n_bits: int, n_samples: int) -> GeneratedDataset:
        if n_bits <= 0 or n_bits % 8 != 0:
            raise ValueError(f"n_bits must be a positive multiple of 8, got {n_bits}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        n_bytes = n_bits // 8
        samples = np.empty((n_samples, n_bytes), dtype=np.uint8)
        seeds = np.empty(n_samples, dtype=np.int64)

        for i in range(n_samples):
            sample = self.generator.generate_sample(n_bytes)
            samples[i] = np.frombuffer(sample.data, dtype=np.uint8)
            seeds[i] = NO_SEED if sample.seed is None else sample.seed

        return GeneratedDataset(
            generator_name=self.generator.name,
            generator_params=self.generator.params(),
            label=self.generator.label,
            n_bits=n_bits,
            samples=samples,
            seeds=seeds,
        )
