"""Usage:
    python main.py generate csprng
    python main.py generate csprng --lengths 128 1024 --samples 10000
    python main.py generate lcg
    python main.py generate lcg --output-shift 0
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
WEAK_ONLY_OPTIONS = ("seed_bits", "output_shift")


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
    generate.add_argument("--output-shift", type=int, default=None,
                          help="state bits dropped before taking the output byte (lcg only); "
                               "0 = raw low byte, the most defective variant")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        options = {name: getattr(args, name) for name in WEAK_ONLY_OPTIONS if getattr(args, name) is not None}
        if options and args.generator != "lcg":
            flags = ", ".join("--" + name.replace("_", "-") for name in options)
            raise SystemExit(f"{flags} only supported by the lcg generator")
        generate_datasets(GENERATORS[args.generator](**options), args.lengths, args.samples, args.output)


if __name__ == "__main__":
    main()
