"""Report figures for the feature-separability stage (IEEE double-column width, light background for print).

Labels are in Polish (the report language), with a decimal comma.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator

from src.analysis.separability import MONOBIT_FEATURE, REFERENCE, display_name, load_features

FIG_WIDTH = 7.16  # inches, IEEE double column
SECURE_COLOR = "#2a78d6"  # categorical slot 1 (blue)
WEAK_COLOR = "#eb6834"  # categorical slot 2 (orange)
MONOBIT_COLOR = "#eb6834"
BEST_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
# Single-hue sequential ramp (blue 100 -> 700): light = chance level, dark = perfect separation.
AUC_CMAP = LinearSegmentedColormap.from_list(
    "auc", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

GROUP_LABELS = {
    "monobit": "monobit",
    "entropy": "entropia",
    "runs": "serie",
    "byte_chi2": "χ² bajtów",
    "autocorrelation": "autokorelacja",
    "bit_planes": "płaszczyzny bitowe",
}
LENGTH_LABEL = "długość ciągu [bity]"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": TEXT_MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.facecolor": "white",
})


def pl(value: float, digits: int) -> str:
    """Polish number format: decimal comma, proper minus sign."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _polish_ticks(axis, digits: int) -> None:
    axis.set_major_formatter(FuncFormatter(lambda value, _: pl(value, digits)))


def _histogram_bins(values: np.ndarray, max_bins: int = 60) -> np.ndarray:
    """Equal-width bins, or one bin per value when the feature is discrete (short sequences)."""
    unique = np.unique(values)
    if len(unique) > max_bins:
        # Autocorrelations take values on a lattice (step 2 / pairs); bin widths that are a multiple
        # of the lattice step avoid the comb pattern of equal-width bins.
        step = np.diff(unique).min()
        width = step * np.ceil((unique[-1] - unique[0]) / (max_bins * step))
        return np.arange(unique[0] - step / 2, unique[-1] + width, width)
    if len(unique) == 1:
        return np.array([unique[0] - 0.5, unique[0] + 0.5])
    mids = (unique[1:] + unique[:-1]) / 2
    return np.concatenate([[2 * unique[0] - mids[0]], mids, [2 * unique[-1] - mids[-1]]])


def _save(fig, output: Path, name: str) -> Path:
    path = output / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def auc_heatmap(table: pd.DataFrame, output: Path) -> Path:
    """Best AUC within each feature group, per weak generator (panel) and sequence length (column)."""
    generators = list(dict.fromkeys(table["generator"]))
    groups = list(dict.fromkeys(table["group"]))
    lengths = sorted(table["n_bits"].unique())
    best = table.groupby(["generator", "group", "n_bits"])["auc"].max()

    fig, axes = plt.subplots(1, len(generators), figsize=(FIG_WIDTH, 0.4 * len(groups) + 1.2),
                             sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, generator in zip(axes, generators):
        grid = np.array([[best[generator, group, n] for n in lengths] for group in groups])
        image = ax.imshow(grid, cmap=AUC_CMAP, vmin=0.5, vmax=1.0, aspect="auto")
        for (i, k), value in np.ndenumerate(grid):
            ax.text(k, i, pl(value, 2), ha="center", va="center", fontsize=6.5,
                    color="white" if value > 0.8 else TEXT)
        ax.set_xticks(range(len(lengths)), [str(n) for n in lengths])
        ax.set_yticks(range(len(groups)), [GROUP_LABELS.get(group, group) for group in groups])
        ax.set_xlabel(LENGTH_LABEL)
        ax.set_title(display_name(generator))
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=axes, label="ROC-AUC (najlepsza cecha w grupie)", shrink=0.9)
    _polish_ticks(colorbar.ax.yaxis, 1)
    return _save(fig, output, "auc_heatmap")


def best_vs_monobit(table: pd.DataFrame, output: Path) -> Path:
    """AUC vs. length: best single feature against the NIST monobit statistic, one panel per weak generator."""
    generators = list(dict.fromkeys(table["generator"]))
    fig, axes = plt.subplots(1, len(generators), figsize=(FIG_WIDTH, 2.2), sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, generator in zip(axes, generators):
        rows = table[table["generator"] == generator]
        best = rows.groupby("n_bits")["auc"].max()
        monobit = rows[rows["feature"] == MONOBIT_FEATURE].set_index("n_bits")["auc"].sort_index()
        ax.plot(best.index, best.values, color=BEST_COLOR, lw=2, marker="o", ms=4, label="najlepsza pojedyncza cecha")
        ax.plot(monobit.index, monobit.values, color=MONOBIT_COLOR, lw=2, marker="s", ms=4,
                label="test monobitowy (NIST)")
        ax.axhline(0.93, color=TEXT_MUTED, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(best.index, [str(n) for n in best.index])
        ax.minorticks_off()
        ax.set_ylim(0.45, 1.03)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(display_name(generator))
        ax.set_xlabel(LENGTH_LABEL)
        _polish_ticks(ax.yaxis, 1)
    axes[0].set_ylabel("ROC-AUC")
    axes[-1].text(0.98, 0.935, "kryterium sukcesu 0,93", transform=axes[-1].get_yaxis_transform(),
                  fontsize=6.5, color=TEXT_MUTED, ha="right", va="bottom")
    axes[-1].legend(loc="center right", frameon=False)
    return _save(fig, output, "auc_vs_length")


def feature_distributions(table: pd.DataFrame, features_root: Path, n_bits: int, output: Path) -> Path:
    """Histograms of each weak generator's most separating feature at one length, CSPRNG overlaid."""
    generators = list(dict.fromkeys(table["generator"]))
    rows = table[table["n_bits"] == n_bits]
    fig, axes = plt.subplots(1, len(generators), figsize=(FIG_WIDTH, 2.2), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, generator in zip(axes, generators):
        top = rows[rows["generator"] == generator].sort_values("auc", ascending=False).iloc[0]
        weak, names, _ = load_features(features_root, generator, n_bits)
        secure, _, _ = load_features(features_root, REFERENCE, n_bits)
        j = names.index(top["feature"])
        bins = _histogram_bins(np.concatenate([weak[:, j], secure[:, j]]))
        ax.hist(secure[:, j], bins=bins, color=SECURE_COLOR, histtype="step", lw=1.5, label="CSPRNG")
        ax.hist(weak[:, j], bins=bins, color=WEAK_COLOR, histtype="step", lw=1.5, label="LCG")
        ax.set_title(f"{display_name(generator)}\n{top['feature']} (AUC {pl(top['auc'], 3)})")
        ax.set_xlabel("wartość cechy")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        _polish_ticks(ax.xaxis, 2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Cecha najlepiej rozróżniająca generatory, n = {n_bits} bitów", fontsize=9)
    return _save(fig, output, f"feature_distributions_n{n_bits}")
