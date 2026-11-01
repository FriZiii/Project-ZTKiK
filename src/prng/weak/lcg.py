from dataclasses import dataclass

# Classic ANSI C / glibc rand() parameters.
DEFAULT_A = 1103515245
DEFAULT_C = 12345
DEFAULT_M = 2**31
DEFAULT_OUTPUT_SHIFT = 16  # rand() exposes the higher bits, low bits of a power-of-two LCG are periodic


@dataclass
class LCG:
    """Linear congruential generator: x_{k+1} = (a * x_k + c) mod m."""

    state: int
    a: int = DEFAULT_A
    c: int = DEFAULT_C
    m: int = DEFAULT_M
    output_shift: int = DEFAULT_OUTPUT_SHIFT

    def next_state(self) -> int:
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_byte(self) -> int:
        return (self.next_state() >> self.output_shift) & 0xFF

    def next_bytes(self, n_bytes: int) -> bytes:
        return bytes(self.next_byte() for _ in range(n_bytes))
