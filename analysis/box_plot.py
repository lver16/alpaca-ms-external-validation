import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# =========================================================
# SETTINGS
# =========================================================

target = "prl" # "lesion" or "prl"

comp = "aprl" # "flames" or "aprl"

PRL = "_PRL_ref"  # "" or "_PRL_pred" or "_PRL_ref" (si PRL != "", utilise que target = "prl")

# ── NEW: set to None to plot both models (original behaviour)
#         set to "alpaca", "flames", or "aprl" to plot a single model
single_model = None   # e.g. "alpaca" | "flames" | "aprl" | None


alpaca_excel = Path(f"output_comp/alpaca_metrics_with_wo_CLU{PRL}/processed/metrics_summary_{target}_comparison.xlsx")

if "flames" in comp:
    comp_excel = Path(f"output_comp/alpaca_metrics_with_wo_CLU_{comp}{PRL}/processed/metrics_summary_{target}_comparison.xlsx")
else: 
    comp_excel = Path(f"output_comp/aprl_metrics_with_wo_CLU_50{PRL}/processed/metrics_summary_{target}_comparison.xlsx")
    
sheet_name = f"lesion_selected_metrics"

# Full dataset metrics
metrics_full = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision"]

# CLU metrics
metrics_clu = ["F1_CLU", "Recall_CLU", "Precision_CLU"]

metrics_all = metrics_full + metrics_clu

# Multiply by 100 for display
as_percentage = True

# Output files
output_dir = Path("boxplot_outputs")
output_dir.mkdir(exist_ok=True)

save_path_full = output_dir / f"boxplots_full_dataset_alpaca_vs_{comp}{PRL}"
save_path_clu  = output_dir / f"boxplots_CLU_alpaca_vs_{comp}{PRL}"
save_path_all  = output_dir / f"boxplots_all_metrics_alpaca_vs_{comp}_{target}_50{PRL}"

# Reproducible jitter
np.random.seed(42)

# =========================================================
# LOAD DATA
# =========================================================
df_alpaca = pd.read_excel(alpaca_excel, sheet_name=sheet_name)
df_comp   = pd.read_excel(comp_excel,   sheet_name=sheet_name)

df_alpaca.columns = [str(c).strip() for c in df_alpaca.columns]
df_comp.columns   = [str(c).strip() for c in df_comp.columns]

# =========================================================
# HELPERS
# =========================================================
def get_available_metrics(df1, df2, metrics):
    if df2 is None:
        return [m for m in metrics if m in df1.columns]
    return [m for m in metrics if m in df1.columns and m in df2.columns]


def add_pvalue(ax, vals1, vals2, y_max, as_pct):
    # p-value requires two groups — skip entirely in single-model mode
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
        return  # ns: don't annotate

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
    df2 = None  →  single-model plot (df1 only).
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

    # Determine labels
    if single:
        if single_model == "alpaca":
            labels = ["ALPaCA"]
            dot_colors = [blue]
        elif single_model == "flames":
            labels = ["FLAMeS"]
            dot_colors = [red]
        else:  # "aprl"
            labels = ["APRL"]
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

        if as_percentage:
            vals1 = vals1 * 100
            if vals2 is not None:
                vals2 = vals2 * 100

        data_to_plot = [vals1] if single else [vals1, vals2]

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

        # Jittered subject-level points
        jitter_strength = 0.06
        for xi, (vals, color) in enumerate(zip(data_to_plot, dot_colors), start=1):
            x_jitter = np.random.normal(loc=xi, scale=jitter_strength, size=len(vals))
            ax.scatter(x_jitter, vals, color=color, alpha=0.45, s=20, zorder=3)
            if len(vals) > 0:
                ax.scatter(xi, vals.mean(), marker="D", s=35, color="red",
                           edgecolor="black", linewidth=0.7, zorder=4)

        metric_title = metric.replace("_CLU", r"$^{CLU}$")
        ax.set_title(metric_title, fontsize=15, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_facecolor("#e6e6e6")

        if as_percentage:
            ax.set_ylabel("%")
            ax.set_ylim(-5, 105)
        else:
            ax.set_ylabel(metric)

        # p-value (skipped automatically when vals2 is None)
        y_max = vals1.max() if len(vals1) > 0 else 0
        if vals2 is not None and len(vals2) > 0:
            y_max = max(y_max, vals2.max())
        add_pvalue(ax, vals1.values, vals2.values if vals2 is not None else None, y_max, as_percentage)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    print(f"Saved figure to: {save_path}")


# =========================================================
# RESOLVE df2 AND TITLES BASED ON single_model
# =========================================================

if single_model is not None:
    # Pick the right dataframe and build a title / save path
    if single_model == "alpaca":
        df1_plot = df_alpaca
        model_label = "ALPaCA"
    elif single_model == "flames":
        df1_plot = df_comp   # flames output is in comp_excel when comp == "flames"
        model_label = "FLAMeS"
    else:  # "aprl"
        df1_plot = df_comp
        model_label = "APRL"

    df2_plot   = None
    tit_all    = f"Full comparison: {model_label}"
    save_path_all_pdf = (output_dir / f"boxplots_all_metrics_{single_model}_{target}{PRL}").with_suffix(".pdf")

else:
    df1_plot = df_alpaca
    df2_plot = df_comp

    if "flames" in comp:
        tit_all = "Full comparison: MIMoSA vs FLAMeS"
    else:
        tit_all = "Full comparison: ALPaCA vs APRL"

    save_path_all_pdf = save_path_all.with_suffix(".pdf")


# =========================================================
# PLOTS
# =========================================================

make_boxplot_figure(
    df1_plot,
    df2_plot,
    metrics_all,
    title=tit_all,
    save_path=save_path_all_pdf,
    comparison=comp,
)