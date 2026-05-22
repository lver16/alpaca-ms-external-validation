"""
roc_plot.py
-------------------------------------------------------------------------------
Description:
    Generates ROC curve figures for lesion and PRL classification from ALPaCA
    predictions. For each task (lesion and PRL), the script:
      1. Loads true labels and predicted scores from an Excel file
      2. Computes the ROC curve and AUROC
      3. Estimates a 95% confidence interval via bootstrapping
      4. Identifies the optimal threshold using Youden's J statistic
      5. Saves one PDF figure per task

    Input Excel file must contain two sheets:
      - "lesion" : columns true_label, Lesion
      - "prl"    : columns true_label, PRL

Usage:
    python roc_plot.py --input <input_file>
                       [--output_lesion <output_lesion>]
                       [--output_prl <output_prl>]

Arguments:
    --input          Path to the input Excel file containing ROC data (required)
    --output_lesion  Output path for the lesion ROC figure
                     (default: roc_lesion.pdf)
    --output_prl     Output path for the PRL ROC figure
                     (default: roc_prl.pdf)

Example:
    python roc_plot.py --input /data/results/roc_data.xlsx
    python roc_plot.py --input /data/results/roc_data.xlsx \\
            --output_lesion /data/figures/roc_lesion.pdf \\
            --output_prl /data/figures/roc_prl.pdf

Outputs:
    - <output_lesion> : ROC curve for lesion classification
    - <output_prl>    : ROC curve for PRL classification

Notes:
    - Bootstrap uses 2000 resamples with fixed seed 42 for reproducibility
    - Youden's J statistic (sensitivity + specificity - 1) is used to select
      the optimal threshold, shown as a red dot on the curve
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Generate ROC curve figures for lesion and PRL classification."
)
parser.add_argument(
    "--input", required=True,
    help="Path to the input Excel file containing ROC data"
)
parser.add_argument(
    "--output_lesion", default="roc_lesion.pdf",
    help="Output path for the lesion ROC figure (default: roc_lesion.pdf)"
)
parser.add_argument(
    "--output_prl", default="roc_prl.pdf",
    help="Output path for the PRL ROC figure (default: roc_prl.pdf)"
)
args = parser.parse_args()

INPUT_FILE    = args.input
OUTPUT_LESION = args.output_lesion
OUTPUT_PRL    = args.output_prl

# -----------------------------------------------------------------------------
# Fixed settings
# -----------------------------------------------------------------------------
N_BOOTSTRAP = 2000   # number of bootstrap resamples for AUROC confidence interval
RANDOM_SEED = 42     # fixed seed for reproducible bootstrap

# Plot colors
COLOR_CURVE  = "#2166AC"   # ROC curve
COLOR_YOUDEN = "#CC0000"   # Youden's J optimal threshold point
COLOR_DIAG   = "#AAAAAA"   # diagonal reference line


# -----------------------------------------------------------------------------
# Bootstrap AUROC confidence interval
# -----------------------------------------------------------------------------

def bootstrap_auroc_ci(y_true, y_score, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Estimate 95% confidence interval for AUROC via percentile bootstrapping.

    Parameters:
        y_true  : binary true labels
        y_score : predicted scores
        n_boot  : number of bootstrap resamples
        seed    : random seed for reproducibility

    Returns:
        (ci_lo, ci_hi) : lower and upper bounds of the 95% CI
    """
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Skip resamples with only one class (cannot compute ROC)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        aucs.append(auc(fpr_b, tpr_b))

    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# -----------------------------------------------------------------------------
# Optimal threshold selection
# -----------------------------------------------------------------------------

def youden_point(fpr, tpr, thresholds):
    """
    Find the optimal threshold using Youden's J statistic (sensitivity + specificity - 1).

    Returns:
        (threshold, sensitivity, specificity) at the optimal point
    """
    idx = np.argmax(tpr - fpr)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]


# -----------------------------------------------------------------------------
# Single panel plotting
# -----------------------------------------------------------------------------

def plot_panel(ax, y_true, y_score, title):
    """
    Draw a single ROC curve panel on the given axes.
    Prints AUROC, 95% CI, optimal threshold, sensitivity, and specificity.
    """
    # Remove missing values
    mask    = ~np.isnan(y_score)
    y_true  = y_true[mask]
    y_score = y_score[mask]

    # Compute ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc                = auc(fpr, tpr)
    ci_lo, ci_hi         = bootstrap_auroc_ci(y_true, y_score)
    thresh, sens, spec   = youden_point(fpr, tpr, thresholds)

    # Drop the last point (FPR=1, TPR=1) appended automatically by sklearn
    fpr = fpr[:-1]
    tpr = tpr[:-1]

    # Draw diagonal reference line, ROC curve, and Youden optimal point
    ax.plot([1, 0], [0, 1], color=COLOR_DIAG, lw=1.2, linestyle="--", zorder=1)
    ax.plot(1 - fpr, tpr, color=COLOR_CURVE, lw=2.2, zorder=3,
            label=f"AUROC = {auroc:.2f}")
    ax.scatter([spec], [sens], color=COLOR_YOUDEN, s=60, zorder=5,
               edgecolors="white", linewidths=0.6)

    # Axis styling
    ax.set_xlim([1.02, -0.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.tick_params(labelsize=9)

    # Print summary statistics
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    print(f"  {title:<8}  AUROC={auroc:.3f}  95%CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"Thresh={thresh:.3f}  Sens={sens:.1%}  Spec={spec:.1%}  "
          f"(n={len(y_true)}, pos={n_pos}, neg={n_neg})")

    return auroc, ci_lo, ci_hi, thresh, sens, spec


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print(f"Loading {INPUT_FILE} ...\n")

    if not pd.io.common.file_exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df_les = pd.read_excel(INPUT_FILE, sheet_name="lesion")
    df_prl = pd.read_excel(INPUT_FILE, sheet_name="prl")

    print("-" * 85)
    print(f"  {'Task':<8}  {'AUROC':>6}  {'95% CI':^15}  "
          f"{'Threshold':>10}  {'Sensitivity':>12}  {'Specificity':>12}")
    print("-" * 85)

    # Lesion ROC figure
    fig_les, ax_les = plt.subplots(figsize=(5, 4.8))
    plot_panel(ax_les, df_les["true_label"].values, df_les["Lesion"].values, "Lesion")
    fig_les.tight_layout()
    fig_les.savefig(OUTPUT_LESION, dpi=180, bbox_inches="tight")
    plt.close(fig_les)

    # PRL ROC figure
    fig_prl, ax_prl = plt.subplots(figsize=(5, 4.8))
    plot_panel(ax_prl, df_prl["true_label"].values, df_prl["PRL"].values, "PRL")
    fig_prl.tight_layout()
    fig_prl.savefig(OUTPUT_PRL, dpi=180, bbox_inches="tight")
    plt.close(fig_prl)

    print("-" * 85)
    print(f"\nSaved -> {OUTPUT_LESION}")
    print(f"Saved -> {OUTPUT_PRL}")


if __name__ == "__main__":
    main()