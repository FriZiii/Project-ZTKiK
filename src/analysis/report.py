from pathlib import Path

from src.analysis.plots import auc_heatmap, best_vs_monobit, feature_distributions
from src.analysis.separability import separability_table


def analyze_features(features_root: str, output: str, distribution_lengths: list[int]) -> None:
    """Write feature_auc.csv and the separability figures to `output`."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = separability_table(features_root)
    csv_path = output_dir / "feature_auc.csv"
    table.sort_values(["generator", "n_bits", "auc"], ascending=[True, True, False]).to_csv(
        csv_path, index=False, float_format="%.4f")
    print(f"{len(table)} feature/length/generator AUCs -> {csv_path}")

    best = table.loc[table.groupby(["generator", "n_bits"])["auc"].idxmax()]
    print(best[["generator", "n_bits", "feature", "auc"]].to_string(index=False, float_format="%.3f"))

    paths = [auc_heatmap(table, output_dir), best_vs_monobit(table, output_dir)]
    paths += [feature_distributions(table, Path(features_root), n, output_dir) for n in distribution_lengths]
    for path in paths:
        print(f"figure -> {path}")
