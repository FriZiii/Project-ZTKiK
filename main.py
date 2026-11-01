"""Usage:
    python main.py generate csprng
    python main.py generate csprng --lengths 128 1024 --samples 10000
    python main.py generate lcg
"""

import argparse

from src.dataset.generate import generate_datasets
from src.prng.secure.csprng import CSPRNGGenerator
from src.prng.weak.time_seeded_lcg import TimeSeededLCGGenerator

GENERATORS = {
    "csprng": CSPRNGGenerator,
    "lcg": TimeSeededLCGGenerator,
}

DEFAULT_LENGTHS = [64, 128, 256, 512, 1024, 4096]
DEFAULT_SAMPLES = 50_000
DEFAULT_OUTPUT = "data/raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate bit sequences and save them to disk")
    generate.add_argument("generator", choices=GENERATORS)
    generate.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS,
                          help="sequence lengths in bits (multiples of 8)")
    generate.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="number of samples per length")
    generate.add_argument("--output", default=DEFAULT_OUTPUT, help="output directory")
    generate.add_argument("--seed-bits", type=int, default=None,
                          help="seed size in bits for weak generators (lcg only)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        options = {}
        if args.seed_bits is not None:
            if args.generator != "lcg":
                raise SystemExit("--seed-bits is only supported by the lcg generator")
            options["seed_bits"] = args.seed_bits
        generate_datasets(GENERATORS[args.generator](**options), args.lengths, args.samples, args.output)


if __name__ == "__main__":
    main()
