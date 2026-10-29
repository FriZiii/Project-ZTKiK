import time

from src.dataset.builder import DatasetBuilder
from src.dataset.storage import DatasetStorage
from src.prng.base import BaseGenerator


def generate_datasets(generator: BaseGenerator, lengths: list[int], n_samples: int, output: str) -> None:
    builder = DatasetBuilder(generator)
    storage = DatasetStorage(output)

    for n_bits in lengths:
        start = time.perf_counter()
        dataset = builder.build(n_bits=n_bits, n_samples=n_samples)
        path = storage.save(dataset)
        ones_ratio = dataset.unpacked_bits().mean()
        print(f"{generator.name} | n={n_bits:>5} bits | {dataset.n_samples} samples | "
              f"ones ratio={ones_ratio:.5f} | {time.perf_counter() - start:.2f}s -> {path}")
