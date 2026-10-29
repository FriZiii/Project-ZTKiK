"""Usage:
    python main.py generate csprng
    python main.py generate csprng --lengths 128 1024 --samples 10000
"""

import argparse

from src.dataset.generate import generate_datasets
from src.prng.secure.csprng import CSPRNGGenerator

GENERATORS = {
    "csprng": CSPRNGGenerator,
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate_datasets(GENERATORS[args.generator](), args.lengths, args.samples, args.output)


if __name__ == "__main__":
    main()
