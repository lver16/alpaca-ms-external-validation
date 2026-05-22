"""
shapiro_wilk.py
-------------------------------------------------------------------------------
Description:
    Runs the Shapiro-Wilk normality test on segmentation metrics across all
    method/target/PRL-suffix combinations. For each combination, the script
    loads the processed metrics Excel file produced by summarize_metrics.py
    and tests whether each metric follows a normal distribution.

    The following combinations are tested:
        Methods  : ALPaCA, APRL, FLAMeS
        Targets  : lesion, prl
        Suffixes : (none), _PRL_pred, _PRL_ref, _50 (APRL only)

    Files that do not exist are skipped with a warning, so the script can be
    run even if only a subset of combinations have been computed.

Usage:
    python shapiro_wilk.py --input_dir <input_dir>
                           [--output <output>]

Arguments:
    --input_dir   Root directory containing the output_comp/ folder
                  (default: current working directory)
    --output      Path to the output Excel file with normality test results
                  (default: normality_check.xlsx)

Example:
    python shapiro_wilk.py --input_dir /data/results
    python shapiro_wilk.py --input_dir /data/results --output /data/results/normality.xlsx

Output:
    An Excel file with one row per (method, metric) combination containing:
      - Method   : method and target label (e.g. ALPaCA_lesion)
      - Metric   : metric name
      - N        : number of subjects
      - W-stat   : Shapiro-Wilk W statistic
      - p-value  : p-value of the test
      - Normal?  : "Yes" if p > 0.05, "No" otherwise

Notes:
    - A minimum of 3 values is required to run the test
    - Results are also printed to the console with a summary
"""

import argparse
import pandas as pd
from pathlib import Path
from scipy import stats

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run Shapiro-Wilk normality test on segmentation metrics."
)
parser.add_argument(
    "--input_dir", default=".",
    help="Root directory containing the output_comp/ folder (default: current directory)"
)
parser.add_argument(
    "--output", default="normality_check.xlsx",
    help="Output Excel file path (default: normality_check.xlsx)"
)
args = parser.parse_args()

input_dir   = Path(args.input_dir)
output_path = Path(args.output)

# -----------------------------------------------------------------------------
# Define metrics and sheet name
# -----------------------------------------------------------------------------
metrics_all = [
    "PQ", "DSC", "nDSC", "F1", "Recall", "Precision",
    "F1_CLU", "Recall_CLU", "Precision_CLU"
]

sheet_name = "lesion_selected_metrics"

# -----------------------------------------------------------------------------
# Build file index from input_dir
# All paths follow the convention:
#   output_comp/<method>_metrics_with_wo_CLU<suffix>/processed/
#       metrics_summary_<target>_comparison.xlsx
# -----------------------------------------------------------------------------
comp = input_dir / "output_comp"

files = {
    # ALPaCA — lesion and PRL
    "ALPaCA_lesion"  : comp / "alpaca_metrics_with_wo_CLU/processed/metrics_summary_lesion_comparison.xlsx",
    "ALPaCA_prl"     : comp / "alpaca_metrics_with_wo_CLU/processed/metrics_summary_prl_comparison.xlsx",
    "ALPaCA_prl_pred": comp / "alpaca_metrics_with_wo_CLU_PRL_pred/processed/metrics_summary_prl_comparison.xlsx",
    "ALPaCA_prl_ref" : comp / "alpaca_metrics_with_wo_CLU_PRL_ref/processed/metrics_summary_prl_comparison.xlsx",

    # APRL — lesion and PRL
    "APRL_lesion"    : comp / "aprl_metrics_with_wo_CLU/processed/metrics_summary_lesion_comparison.xlsx",
    "APRL_prl"       : comp / "aprl_metrics_with_wo_CLU/processed/metrics_summary_prl_comparison.xlsx",
    "APRL_prl_pred"  : comp / "aprl_metrics_with_wo_CLU_PRL_pred/processed/metrics_summary_prl_comparison.xlsx",
    "APRL_prl_ref"   : comp / "aprl_metrics_with_wo_CLU_PRL_ref/processed/metrics_summary_prl_comparison.xlsx",
    "APRL_lesion_05" : comp / "aprl_metrics_with_wo_CLU_50/processed/metrics_summary_lesion_comparison.xlsx",
    "APRL_prl_05"    : comp / "aprl_metrics_with_wo_CLU_50/processed/metrics_summary_prl_comparison.xlsx",

    # FLAMeS — lesion and PRL
    "FLAMeS_lesion"  : comp / "alpaca_metrics_with_wo_CLU_flames_05_thresh/processed/metrics_summary_lesion_comparison.xlsx",
    "FLAMeS_prl"     : comp / "alpaca_metrics_with_wo_CLU_flames_05_thresh/processed/metrics_summary_prl_comparison.xlsx",
    "FLAMeS_prl_pred": comp / "alpaca_metrics_with_wo_CLU_flames_05_thresh_PRL_pred/processed/metrics_summary_prl_comparison.xlsx",
    "FLAMeS_prl_ref" : comp / "alpaca_metrics_with_wo_CLU_flames_05_thresh_PRL_ref/processed/metrics_summary_prl_comparison.xlsx",
}

# -----------------------------------------------------------------------------
# Run Shapiro-Wilk test for each method/metric combination
# -----------------------------------------------------------------------------
rows = []

for label, path in files.items():
    if not path.exists():
        print(f"[WARNING] File not found, skipping: {path}")
        continue

    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    for metric in metrics_all:
        if metric not in df.columns:
            print(f"[WARNING] Metric '{metric}' not found in {label}, skipping.")
            continue

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()

        if len(vals) < 3:
            print(f"[WARNING] Not enough values for '{metric}' in {label} (n={len(vals)}), skipping.")
            continue

        stat, p = stats.shapiro(vals)

        rows.append({
            "Method" : label,
            "Metric" : metric,
            "N"      : len(vals),
            "W-stat" : round(stat, 4),
            "p-value": round(p, 4),
            "Normal?": "Yes" if p > 0.05 else "No"
        })

# -----------------------------------------------------------------------------
# Print results and summary
# -----------------------------------------------------------------------------
df_result = pd.DataFrame(rows)

print("\n===== SHAPIRO-WILK NORMALITY TEST =====\n")
print(df_result.to_string(index=False))

n_total  = len(df_result)
n_normal = (df_result["Normal?"] == "Yes").sum()
n_not    = (df_result["Normal?"] == "No").sum()

print(f"\n--- Summary ---")
print(f"Total tests : {n_total}")
print(f"Normal      : {n_normal} ({100 * n_normal / n_total:.1f}%)")
print(f"Non-normal  : {n_not} ({100 * n_not / n_total:.1f}%)")

# -----------------------------------------------------------------------------
# Save results to Excel
# -----------------------------------------------------------------------------
df_result.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")