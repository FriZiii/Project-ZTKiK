"""Usage:
    python main.py generate csprng
    python main.py generate csprng --lengths 128 1024 --samples 10000
    python main.py generate lcg
    python main.py generate lcg --output-shift 0
    python main.py features
    python main.py features --generators csprng lcg --lengths 1024 --groups monobit entropy
    python main.py analyze
"""

import argparse

from src.analysis.report import analyze_features
from src.dataset.generate import generate_datasets
from src.features.compute import compute_features
from src.features.extract import FEATURE_GROUPS
from src.prng.secure.csprng import CSPRNGGenerator
from src.prng.weak.time_seeded_lcg import TimeSeededLCGGenerator

GENERATORS = {
    "csprng": CSPRNGGenerator,
    "lcg": TimeSeededLCGGenerator,
}

DEFAULT_LENGTHS = [64, 128, 256, 512, 1024, 4096]
DEFAULT_SAMPLES = 50_000
DEFAULT_OUTPUT = "data/raw"
DEFAULT_FEATURES_OUTPUT = "data/features"
DEFAULT_REPORT_OUTPUT = "reports/figures"
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

    features = commands.add_parser("features", help="extract statistical features from generated datasets")
    features.add_argument("--input", default=DEFAULT_OUTPUT, help="directory with generated datasets")
    features.add_argument("--output", default=DEFAULT_FEATURES_OUTPUT, help="output directory")
    features.add_argument("--generators", nargs="+", default=None, help="dataset names (default: all found)")
    features.add_argument("--lengths", type=int, nargs="+", default=None, help="lengths in bits (default: all)")
    features.add_argument("--groups", nargs="+", choices=FEATURE_GROUPS, default=None,
                          help="feature groups (default: all)")

    analyze = commands.add_parser("analyze", help="per-feature ROC-AUC table and report figures")
    analyze.add_argument("--input", default=DEFAULT_FEATURES_OUTPUT, help="directory with extracted features")
    analyze.add_argument("--output", default=DEFAULT_REPORT_OUTPUT, help="output directory for CSV and figures")
    analyze.add_argument("--distribution-lengths", type=int, nargs="+", default=[256, 4096],
                         help="lengths for which feature histograms are drawn")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        options = {name: getattr(args, name) for name in WEAK_ONLY_OPTIONS if getattr(args, name) is not None}
        if options and args.generator != "lcg":
            flags = ", ".join("--" + name.replace("_", "-") for name in options)
            raise SystemExit(f"{flags} only supported by the lcg generator")
        generate_datasets(GENERATORS[args.generator](**options), args.lengths, args.samples, args.output)
    elif args.command == "features":
        compute_features(args.input, args.output, args.generators, args.lengths, args.groups)
    elif args.command == "analyze":
        analyze_features(args.input, args.output, args.distribution_lengths)


if __name__ == "__main__":
    main()
