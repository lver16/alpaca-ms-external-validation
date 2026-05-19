"""
bland_altman_plot.py

Generates two separate Bland-Altman figures, one per model (A and B).
By default only the Lesion Count row is shown; pass --include_clu
to also display the CLU Count row below within each figure.

Layout per figure (default, lesion count only):
    [Model - Lesion Count]

Layout per figure (with --include_clu):
    [Model - Lesion Count]
    [Model - CLU Count   ]

Output files:
    <output_a>   (default: bland_altman_a.png)
    <output_b>   (default: bland_altman_b.png)

Usage:
    # Lesion count only (default)
    python bland_altman_plot.py \
        --input_a metrics_alpaca.xlsx --label_a ALPaCA \
        --input_b metrics_flames.xlsx --label_b FLaMeS

    # With CLU row
    python bland_altman_plot.py \
        --input_a metrics_alpaca.xlsx --label_a ALPaCA \
        --input_b metrics_flames.xlsx --label_b FLaMeS \
        --include_clu

    # Custom output names
    python bland_altman_plot.py \
        --input_a metrics_alpaca.xlsx --label_a ALPaCA \
        --input_b metrics_flames.xlsx --label_b FLaMeS \
        --output_a alpaca_ba.png --output_b flames_ba.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats


def bland_altman_plot(ax, method1, method2, label1, label2, title,
                      color='#2E86AB', xlim=None, ylim=None):
    """
    Draw a Bland-Altman plot on the given axes.
    method1: reference (ground truth)
    method2: predicted
    """
    mean = (method1 + method2) / 2
    diff = method2 - method1        # predicted - reference

    md    = np.mean(diff)
    sd    = np.std(diff, ddof=1)
    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd

    n      = len(diff)
    se_md  = sd / np.sqrt(n)
    se_loa = np.sqrt(3 * sd**2 / n)
    ci_md  = 1.96 * se_md
    ci_loa = 1.96 * se_loa

    # R² and p-value: proportional bias test
    r, p_value = stats.pearsonr(mean, diff)
    r2    = r ** 2
    p_str = f'{p_value:.3f}' if p_value >= 0.001 else '< 0.001'

    # ── scatter ───────────────────────────────────────────────────────────────
    ax.scatter(mean, diff, color=color, alpha=0.7, edgecolors='white',
               linewidths=0.5, s=60, zorder=3)

    pad = (mean.max() - mean.min()) * 0.05
    x0, x1 = mean.min() - pad, mean.max() + pad

    # ── reference lines ───────────────────────────────────────────────────────
    ax.axhline(md,    color='#E84855', linewidth=2,   linestyle='-',  zorder=2, label='Mean')
    ax.axhline(loa_u, color='#F4A261', linewidth=1.5, linestyle='--', zorder=2, label='+1.96 SD')
    ax.axhline(loa_l, color='#F4A261', linewidth=1.5, linestyle='--', zorder=2, label='−1.96 SD')
    ax.axhline(0,     color='grey',    linewidth=1,   linestyle=':',  zorder=1)

    # ── inline labels ─────────────────────────────────────────────────────────
    y_range = loa_u - loa_l
    offset  = y_range * 0.03

    kw = dict(fontsize=8, fontweight='bold', zorder=5)
    ax.text(0.98, md    + offset, f'Mean = {md:.2f}',
            color='#E84855', va='bottom', ha='right',
            transform=ax.get_yaxis_transform(), **kw)
    ax.text(0.98, loa_u + offset, f'+1.96 SD = {loa_u:.2f}',
            color='#F4A261', va='bottom', ha='right',
            transform=ax.get_yaxis_transform(), **kw)
    ax.text(0.98, loa_l + offset, f'−1.96 SD = {loa_l:.2f}',
            color='#F4A261', va='bottom', ha='right',
            transform=ax.get_yaxis_transform(), **kw)

    # ── style ─────────────────────────────────────────────────────────────────
    ax.set_xlim(xlim if xlim is not None else (x0, x1))
    ax.set_ylim(ylim)
    ax.set_xlabel(f'Mean of {label1} and {label2}', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'{label2} − {label1}', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_facecolor('#FFFFFF')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # ── stats box ─────────────────────────────────────────────────────────────
    stats_txt = (f'n = {n}\n'
                 f'R² = {r2:.3f}  (p {p_str})')
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes,
            va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F4F8',
                      edgecolor='#CCCCCC', alpha=0.9))

    ax.legend(fontsize=8, loc='lower right', framealpha=0.8)

    return md, sd, loa_l, loa_u


def load_and_check(input_file, need_clu=False):
    df = pd.read_excel(input_file)
    required = ['Pred_Lesion_Count', 'Ref_Lesion_Count']
    if need_clu:
        required += ['CLU_Count']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_file}. "
                             f"Available: {df.columns.tolist()}")
    return df


def plot_one_model(df, label, output_file, include_clu=False):
    """Build and save a single-model Bland-Altman figure."""
    n_rows = 2 if include_clu else 1
    fig_h  = 12 if include_clu else 6

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, fig_h))
    fig.patch.set_facecolor('#F8F9FA')

    # Normalise to always be a list so indexing is consistent
    if n_rows == 1:
        axes = [axes]

    # ── Row 0: Lesion counts ──────────────────────────────────────────────────
    bland_altman_plot(
        axes[0],
        method1=df['Ref_Lesion_Count'].values.astype(float),
        method2=df['Pred_Lesion_Count'].values.astype(float),
        label1='Reference', label2='Predicted',
        title=f'{label} — Lesion Count',
        color='#2E86AB',
        xlim=(-10, 110),
        ylim=(-50, 130),
    )

    # ── Row 1: CLU counts (optional) ──────────────────────────────────────────
    if include_clu:
        bland_altman_plot(
            axes[1],
            method1=df['Ref_Lesion_Count'].values.astype(float),
            method2=df['CLU_Count'].values.astype(float),
            label1='Ref Lesion Count', label2='CLU Count',
            title=f'{label} — CLU Count',
            color='#E84855',
            xlim=(-5, 45),
            ylim=(-40, 20),
        )

    fig.suptitle(f'Bland-Altman Analysis — {label}',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_file}")


def main(input_a, label_a, output_a,
         input_b, label_b, output_b,
         include_clu=False):

    df_a = load_and_check(input_a, need_clu=include_clu)
    df_b = load_and_check(input_b, need_clu=include_clu)

    plot_one_model(df_a, label_a, output_a, include_clu=include_clu)
    plot_one_model(df_b, label_b, output_b, include_clu=include_clu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Separate Bland-Altman plots for two models.')
    parser.add_argument('--input_a',     required=True,
                        help='Excel metrics file for model A (e.g. ALPaCA)')
    parser.add_argument('--label_a',     default='Model A',
                        help='Display name for model A (default: Model A)')
    parser.add_argument('--output_a',    default='bland_altman_a.png',
                        help='Output image for model A (default: bland_altman_a.png)')
    parser.add_argument('--input_b',     required=True,
                        help='Excel metrics file for model B (e.g. FLaMeS)')
    parser.add_argument('--label_b',     default='Model B',
                        help='Display name for model B (default: Model B)')
    parser.add_argument('--output_b',    default='bland_altman_b.png',
                        help='Output image for model B (default: bland_altman_b.png)')
    parser.add_argument('--include_clu', action='store_true',
                        help='Add a CLU Count row to each figure')
    args = parser.parse_args()

    main(
        args.input_a, args.label_a, args.output_a,
        args.input_b, args.label_b, args.output_b,
        include_clu=args.include_clu,
    )