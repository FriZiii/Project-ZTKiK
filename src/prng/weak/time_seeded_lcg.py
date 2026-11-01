import time

from src.prng.base import BaseGenerator, Sample
from src.prng.weak.lcg import DEFAULT_A, DEFAULT_C, DEFAULT_M, DEFAULT_OUTPUT_SHIFT, LCG

# 2**24 seeds is trivially brute-forceable, yet large enough that 50k samples rarely share a seed
# (with 16 bits ~30% of samples would be duplicates, leaking between train and test splits).
DEFAULT_SEED_BITS = 24


class TimeSeededLCGGenerator(BaseGenerator):
    """Weak PRNG typical for cheap IoT firmware: an LCG seeded with the current time.

    The seed is a high-resolution tick counter truncated to `seed_bits` bits,
    so the whole seed space has only 2**seed_bits values and is guessable by an attacker.
    """

    name = "lcg"
    label = 1

    def __init__(self, seed_bits: int = DEFAULT_SEED_BITS, a: int = DEFAULT_A, c: int = DEFAULT_C,
                 m: int = DEFAULT_M, output_shift: int = DEFAULT_OUTPUT_SHIFT):
        if not 1 <= seed_bits <= 63:
            raise ValueError(f"seed_bits must be in [1, 63], got {seed_bits}")
        self.seed_bits = seed_bits
        self.a, self.c, self.m, self.output_shift = a, c, m, output_shift

    def _time_seed(self) -> int:
        # time.time_ns() has ~15.6 ms resolution on Windows, which would give every sample in a tight
        # loop the same seed; the performance counter (100 ns ticks) mimics a hardware tick counter.
        return (time.perf_counter_ns() // 100) & ((1 << self.seed_bits) - 1)

    def generate_sample(self, n_bytes: int) -> Sample:
        seed = self._time_seed()
        lcg = LCG(state=seed, a=self.a, c=self.c, m=self.m, output_shift=self.output_shift)
        return Sample(data=lcg.next_bytes(n_bytes), seed=seed)

    def params(self) -> dict:
        return {
            "algorithm": "LCG x_{k+1} = (a * x_k + c) mod m",
            "a": self.a,
            "c": self.c,
            "m": self.m,
            "output": f"(state >> {self.output_shift}) & 0xFF per step",
            "seed_source": "time.perf_counter_ns() // 100 (100 ns ticks)",
            "seed_bits": self.seed_bits,
        }
