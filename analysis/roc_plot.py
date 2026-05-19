import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

INPUT_FILE    = "output_comp/roc_data.xlsx"
OUTPUT_LESION = "roc_lesion.pdf"
OUTPUT_PRL    = "roc_prl.pdf"
N_BOOTSTRAP   = 2000
RANDOM_SEED   = 42

COLOR_CURVE  = "#2166AC"
COLOR_YOUDEN = "#CC0000"
COLOR_DIAG   = "#AAAAAA"


def bootstrap_auroc_ci(y_true, y_score, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng  = np.random.default_rng(seed)
    aucs = []
    n    = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        aucs.append(auc(fpr_b, tpr_b))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def youden_point(fpr, tpr, thresholds):
    idx = np.argmax(tpr - fpr)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]


def plot_panel(ax, y_true, y_score, title):
    mask    = ~np.isnan(y_score)
    y_true  = y_true[mask]
    y_score = y_score[mask]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc                = auc(fpr, tpr)
    ci_lo, ci_hi         = bootstrap_auroc_ci(y_true, y_score)
    thresh, sens, spec   = youden_point(fpr, tpr, thresholds)

    # Drop the last point (FPR=1, TPR=1) that sklearn always appends
    fpr = fpr[:-1]
    tpr = tpr[:-1]

    ax.plot([1, 0], [0, 1], color=COLOR_DIAG, lw=1.2, linestyle="--", zorder=1)
    ax.plot(1 - fpr, tpr, color=COLOR_CURVE, lw=2.2, zorder=3,
            label=f"AUROC = {auroc:.2f}")
    ax.scatter([spec], [sens], color=COLOR_YOUDEN, s=60, zorder=5,
               edgecolors="white", linewidths=0.6)

    ax.set_xlim([1.02, -0.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.tick_params(labelsize=9)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    print(f"  {title:<8}  AUROC={auroc:.3f}  95%CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"Thresh={thresh:.3f}  Sens={sens:.1%}  Spec={spec:.1%}  "
          f"(n={len(y_true)}, pos={n_pos}, neg={n_neg})")

    return auroc, ci_lo, ci_hi, thresh, sens, spec


def main():
    print(f"Loading {INPUT_FILE} ...\n")
    df_les = pd.read_excel(INPUT_FILE, sheet_name="lesion")
    df_prl = pd.read_excel(INPUT_FILE, sheet_name="prl")

    print("-" * 85)
    print(f"  {'Task':<8}  {'AUROC':>6}  {'95% CI':^15}  "
          f"{'Threshold':>10}  {'Sensitivity':>12}  {'Specificity':>12}")
    print("-" * 85)

    # ── Lesion figure ─────────────────────────────────────────────────────────
    fig_les, ax_les = plt.subplots(figsize=(5, 4.8))
    plot_panel(ax_les, df_les["true_label"].values, df_les["Lesion"].values, "Lesion")
    fig_les.tight_layout()
    fig_les.savefig(OUTPUT_LESION, dpi=180, bbox_inches="tight")
    plt.close(fig_les)

    # ── PRL figure ────────────────────────────────────────────────────────────
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