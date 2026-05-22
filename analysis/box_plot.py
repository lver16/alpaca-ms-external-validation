"""
box_plot.py
-------------------------------------------------------------------------------
Description:
    Generates boxplots comparing segmentation metrics between two models
    (ALPaCA vs APRL, or MIMoSA vs FLaMeS), or for a single model alone.
    Each metric is displayed as a separate subplot with jittered subject-level
    points, mean diamond markers, and Wilcoxon/Mann-Whitney p-value annotations
    when comparing two models.

    Input Excel files are expected to be the outputs of summarize_metrics.py,
    specifically the sheet "lesion_selected_metrics".

Usage:
    # Compare two models (ALPaCA vs APRL)
    python box_plot.py --target prl --comp aprl --prl_suffix _PRL_ref

    # Compare two models (MIMoSA vs FLaMeS)
    python box_plot.py --target lesion --comp flames --prl_suffix ""

    # Single model only
    python box_plot.py --target prl --comp aprl --prl_suffix _PRL_ref \\
            --single_model alpaca

    # Custom input directory
    python box_plot.py --target prl --comp aprl --prl_suffix _PRL_ref \\
            --input_dir /data/results

Arguments:
    --target        Target type: "lesion" or "prl" (required)
    --comp          Comparison model: "flames" or "aprl" (required)
    --prl_suffix    PRL suffix used in folder names:
                    "" , "_PRL_pred", or "_PRL_ref" (default: "_PRL_ref")
    --single_model  Plot a single model instead of comparing two:
                    "alpaca", "flames", or "aprl" (default: None, plots both)
    --input_dir     Root directory containing the output_comp/ folder
                    (default: current working directory)
    --output_dir    Directory where output figures will be saved
                    (default: boxplot_outputs/)

Notes:
    - Metrics are always displayed as percentages (multiplied by 100)
    - Random seed is fixed at 42 for reproducible jitter
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate boxplots comparing segmentation metrics between two models."
)
parser.add_argument(
    "--target", required=True,
    choices=["lesion", "prl"],
    help="Target type: 'lesion' or 'prl'"
)
parser.add_argument(
    "--comp", required=True,
    choices=["flames", "aprl"],
    help="Comparison model: 'flames' or 'aprl'"
)
parser.add_argument(
    "--prl_suffix", default="_PRL_ref",
    choices=["", "_PRL_pred", "_PRL_ref"],
    help="PRL suffix used in folder names (default: '_PRL_ref')"
)
parser.add_argument(
    "--single_model", default=None,
    choices=["alpaca", "flames", "aprl"],
    help="Plot a single model only instead of comparing two (default: None)"
)
parser.add_argument(
    "--input_dir", default=".",
    help="Root directory containing the output_comp/ folder (default: current directory)"
)
parser.add_argument(
    "--output_dir", default="boxplot_outputs",
    help="Directory where output figures will be saved (default: boxplot_outputs/)"
)
args = parser.parse_args()

target       = args.target
comp         = args.comp
PRL          = args.prl_suffix
single_model = args.single_model
input_dir    = Path(args.input_dir)
output_dir   = Path(args.output_dir)

# -----------------------------------------------------------------------------
# Fixed settings
# -----------------------------------------------------------------------------
as_percentage = True   # metrics are always displayed as percentages
np.random.seed(42)     # fixed seed for reproducible jitter

# -----------------------------------------------------------------------------
# Build input file paths from arguments
# -----------------------------------------------------------------------------
alpaca_excel = input_dir / f"output_comp/alpaca_metrics_with_wo_CLU{PRL}/processed/metrics_summary_{target}_comparison.xlsx"

if "flames" in comp:
    comp_excel = input_dir / f"output_comp/alpaca_metrics_with_wo_CLU_{comp}{PRL}/processed/metrics_summary_{target}_comparison.xlsx"
else:
    comp_excel = input_dir / f"output_comp/aprl_metrics_with_wo_CLU_50{PRL}/processed/metrics_summary_{target}_comparison.xlsx"

sheet_name = "lesion_selected_metrics"

# Validate input files
for path, name in [(alpaca_excel, "alpaca_excel"), (comp_excel, "comp_excel")]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path} ({name})")

output_dir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Define metrics to plot
# -----------------------------------------------------------------------------

# Full dataset metrics
metrics_full = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision"]

# CLU metrics
metrics_clu  = ["F1_CLU", "Recall_CLU", "Precision_CLU"]

metrics_all  = metrics_full + metrics_clu

# -----------------------------------------------------------------------------
# Build output file paths
# -----------------------------------------------------------------------------
save_path_all = output_dir / f"boxplots_all_metrics_alpaca_vs_{comp}_{target}_50{PRL}.pdf"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df_alpaca = pd.read_excel(alpaca_excel, sheet_name=sheet_name)
df_comp   = pd.read_excel(comp_excel,   sheet_name=sheet_name)

df_alpaca.columns = [str(c).strip() for c in df_alpaca.columns]
df_comp.columns   = [str(c).strip() for c in df_comp.columns]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_available_metrics(df1, df2, metrics):
    """Return metrics present in both dataframes (or just df1 in single-model mode)."""
    if df2 is None:
        return [m for m in metrics if m in df1.columns]
    return [m for m in metrics if m in df1.columns and m in df2.columns]


def add_pvalue(ax, vals1, vals2, y_max, as_pct):
    """
    Annotate the plot with a significance marker (* / ** / ***).
    Skipped automatically in single-model mode (vals2 is None).
    Uses Wilcoxon signed-rank test for paired samples, Mann-Whitney otherwise.
    """
    if vals2 is None:
        return

    try:
        if len(vals1) == len(vals2):
            _, p = stats.wilcoxon(vals1, vals2)
        else:
            _, p = stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
    except Exception:
        return

    if p < 0.001:
        p_str = "***"
    elif p < 0.01:
        p_str = "**"
    elif p < 0.05:
        p_str = "*"
    else:
        return  # not significant: skip annotation

    y_min    = min(vals1.min() if len(vals1) > 0 else 0,
                   vals2.min() if len(vals2) > 0 else 0)
    y_offset = (3 if as_pct else abs(y_min) * 0.05 + 1)
    y_line   = y_min - y_offset
    y_text   = y_line - (1 if as_pct else abs(y_min) * 0.02 + 0.5)

    ax.plot([1, 1, 2, 2], [y_line + y_offset * 0.3, y_line, y_line, y_line + y_offset * 0.3],
            color="black", linewidth=0.9)
    ax.text(1.5, y_text, p_str, ha="center", va="top", fontsize=8.5,
            color="black", fontweight="bold")

    current_ylim = ax.get_ylim()
    ax.set_ylim(y_text - (3 if as_pct else 1), current_ylim[1])


def make_boxplot_figure(df1, df2, metrics, title, save_path, comparison):
    """
    Build and save a multi-panel boxplot figure.
    df2=None triggers single-model mode (df1 only, no p-values).
    """
    single = df2 is None
    available_metrics = get_available_metrics(df1, df2, metrics)

    if not available_metrics:
        print(f"[WARNING] No common metrics found for: {title}")
        return

    if not single:
        missing_df1 = [m for m in metrics if m not in df1.columns]
        missing_df2 = [m for m in metrics if m not in df2.columns]
        if missing_df1:
            print(f"[WARNING] Missing in ALPaCA file for '{title}': {missing_df1}")
        if missing_df2:
            print(f"[WARNING] Missing in {comparison} file for '{title}': {missing_df2}")

    n_metrics = len(available_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    blue = "#4C9ED9"
    red  = "#D94B4B"

    # Determine model labels and dot colors
    if single:
        if single_model == "alpaca":
            labels     = ["ALPaCA"]
            dot_colors = [blue]
        elif single_model == "flames":
            labels     = ["FLAMeS"]
            dot_colors = [red]
        else:
            labels     = ["APRL"]
            dot_colors = [red]
    else:
        if "flames" in comparison:
            labels = ["MIMoSA", "FLAMeS"]
        else:
            labels = ["ALPaCA", "APRL"]
        dot_colors = [blue, red]

    for i, metric in enumerate(available_metrics):
        ax = axes[i]

        vals1 = pd.to_numeric(df1[metric], errors="coerce").dropna()
        vals2 = pd.to_numeric(df2[metric], errors="coerce").dropna() if not single else None

        # Convert to percentage
        if as_percentage:
            vals1 = vals1 * 100
            if vals2 is not None:
                vals2 = vals2 * 100

        data_to_plot = [vals1] if single else [vals1, vals2]

        # Draw boxplots
        box = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            boxprops=dict(color="black", linewidth=1.1),
        )
        for b in box["boxes"]:
            b.set_facecolor("white")

        # Jittered subject-level points and mean diamond markers
        jitter_strength = 0.06
        for xi, (vals, color) in enumerate(zip(data_to_plot, dot_colors), start=1):
            x_jitter = np.random.normal(loc=xi, scale=jitter_strength, size=len(vals))
            ax.scatter(x_jitter, vals, color=color, alpha=0.45, s=20, zorder=3)
            if len(vals) > 0:
                ax.scatter(xi, vals.mean(), marker="D", s=35, color="red",
                           edgecolor="black", linewidth=0.7, zorder=4)

        # Axis styling
        metric_title = metric.replace("_CLU", r"$^{CLU}$")
        ax.set_title(metric_title, fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#e6e6e6")

        if as_percentage:
            ax.set_ylabel("%")
            ax.set_ylim(-5, 105)
        else:
            ax.set_ylabel(metric)

        # p-value annotation (skipped automatically in single-model mode)
        y_max = vals1.max() if len(vals1) > 0 else 0
        if vals2 is not None and len(vals2) > 0:
            y_max = max(y_max, vals2.max())
        add_pvalue(ax, vals1.values,
                   vals2.values if vals2 is not None else None,
                   y_max, as_percentage)

    # Remove unused subplot panels
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")


# -----------------------------------------------------------------------------
# Resolve dataframes and title based on single_model argument
# -----------------------------------------------------------------------------
if single_model is not None:
    if single_model == "alpaca":
        df1_plot    = df_alpaca
        model_label = "ALPaCA"
    elif single_model == "flames":
        df1_plot    = df_comp
        model_label = "FLAMeS"
    else:
        df1_plot    = df_comp
        model_label = "APRL"

    df2_plot  = None
    tit_all   = f"Full comparison: {model_label}"
    save_path_all = (output_dir / f"boxplots_all_metrics_{single_model}_{target}{PRL}").with_suffix(".pdf")

else:
    df1_plot = df_alpaca
    df2_plot = df_comp

    if "flames" in comp:
        tit_all = "Full comparison: MIMoSA vs FLAMeS"
    else:
        tit_all = "Full comparison: ALPaCA vs APRL"

# -----------------------------------------------------------------------------
# Generate plot
# -----------------------------------------------------------------------------
make_boxplot_figure(
    df1_plot,
    df2_plot,
    metrics_all,
    title=tit_all,
    save_path=save_path_all,
    comparison=comp,
)