"""
summarize_metrics.py
-------------------------------------------------------------------------------
Description:
    Processes ALPaCA or APRL segmentation metrics from Excel files and produces
    a summary Excel workbook containing per-subject metrics, summary statistics
    (mean, std, min, max), total counts, and a comparison table across Normal,
    CLU, and optionally No-CLU configurations.

    Input files are expected to follow the naming convention:
        output_comp/<method_prl>_metrics_with_wo_CLU<segmentation><PRL>/metrics_<target>.xlsx
        output_comp/<method_prl>_metrics_with_wo_CLU<segmentation><PRL>/metrics_<target>_no_clu.xlsx  (optional)

    Output is saved to:
        output_comp/<method_prl>_metrics_with_wo_CLU<segmentation><PRL>/processed/
            metrics_summary_<target>_comparison.xlsx

Usage:
    python summarize_metrics.py --target <target> --method_prl <method_prl>
                                [--segmentation <segmentation>]
                                [--prl_suffix <prl_suffix>]
                                [--compute_no_clu]
                                [--input_dir <input_dir>]

Arguments:
    --target          Target type to process: "lesion" or "prl" (required)
    --method_prl      Method used for PRL detection: "aprl" or "alpaca" (required)
    --segmentation    Segmentation suffix used in folder name:
                      "" (default), "_flames", or "_flames_05_thresh"
    --prl_suffix      PRL reference/prediction suffix used in folder name:
                      "_50_PRL_ref" (default), "" , or "_PRL_pred"
    --compute_no_clu  Flag to also process the no-CLU metrics file
                      (default: False, only process if this flag is passed)
    --input_dir       Root directory containing the output_comp/ folder
                      (default: current working directory)

Example:
    python summarize_metrics.py --target prl --method_prl aprl
    python summarize_metrics.py --target lesion --method_prl alpaca \\
            --segmentation _flames --prl_suffix _PRL_pred --compute_no_clu
    python summarize_metrics.py --target prl --method_prl aprl \\
            --input_dir /data/results

Output sheets in the Excel workbook:
    - lesion_selected_metrics  : per-subject metrics (full file)
    - lesion_summary_stats     : mean, std, min, max per metric
    - lesion_total_counts      : total TP, FP, FN counts
    - lesion_no_clu_selected   : per-subject metrics (no-CLU file, if computed)
    - lesion_no_clu_summary    : summary stats for no-CLU file (if computed)
    - lesion_no_clu_totals     : total counts for no-CLU file (if computed)
    - comparison_by_type       : comparison table across Normal, CLU, No-CLU
"""

import argparse
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Summarize ALPaCA/APRL segmentation metrics into an Excel workbook."
)
parser.add_argument(
    "--target",
    required=True,
    choices=["lesion", "prl"],
    help="Target type to process: 'lesion' or 'prl'"
)
parser.add_argument(
    "--method_prl",
    required=True,
    choices=["aprl", "alpaca"],
    help="Method used for PRL detection: 'aprl' or 'alpaca'"
)
parser.add_argument(
    "--segmentation",
    default="",
    choices=["", "_flames", "_flames_05_thresh"],
    help="Segmentation suffix used in folder name (default: '')"
)
parser.add_argument(
    "--prl_suffix",
    default="_50_PRL_ref",
    choices=["", "_PRL_pred", "_50_PRL_ref"],
    help="PRL suffix used in folder name (default: '_50_PRL_ref')"
)
parser.add_argument(
    "--compute_no_clu",
    action="store_true",
    help="Also process the no-CLU metrics file (default: False)"
)
parser.add_argument(
    "--input_dir",
    default=".",
    help="Root directory containing the output_comp/ folder (default: current directory)"
)
args = parser.parse_args()

target        = args.target
method_prl    = args.method_prl
segmentation  = args.segmentation
PRL           = args.prl_suffix
COMPUTE_NO_CLU = args.compute_no_clu
input_dir     = Path(args.input_dir)

# -----------------------------------------------------------------------------
# Build input and output paths from arguments
# -----------------------------------------------------------------------------
folder_name = f"{method_prl}_metrics_with_wo_CLU{segmentation}{PRL}"

input_file_full = input_dir / "output_comp" / folder_name / f"metrics_{target}.xlsx"
if COMPUTE_NO_CLU:
    input_file_no_clu = input_dir / "output_comp" / folder_name / f"metrics_{target}_no_clu.xlsx"

output_dir = input_dir / "output_comp" / folder_name / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_excel = output_dir / f"metrics_summary_{target}_comparison.xlsx"

# Check input files exist
if not input_file_full.exists():
    raise FileNotFoundError(f"Input file not found: {input_file_full}")
if COMPUTE_NO_CLU and not input_file_no_clu.exists():
    raise FileNotFoundError(f"No-CLU input file not found: {input_file_no_clu}")

# -----------------------------------------------------------------------------
# Define metrics to process
# -----------------------------------------------------------------------------

# Metrics for the full lesion file (includes CLU variants)
metrics_full = [
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN",
    "Precision_CLU", "Recall_CLU", "F1_CLU", "PQ_CLU"
]

# Metrics for the no-CLU file (no CLU variants)
metrics_no_clu = [
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN"
]

# Metrics displayed as percentages in summary
percentage_metrics = {
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC",
    "Precision_CLU", "Recall_CLU", "F1_CLU", "PQ_CLU"
}

# Metrics shown with std in comparison sheet
metrics_with_std = {"Precision", "Recall", "F1", "PQ", "DSC", "nDSC"}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def process_file(input_file: Path, metrics_to_keep: list, label: str):
    """
    Load an Excel file, compute summary statistics per metric, and return
    the per-subject dataframe, summary stats, and total counts.
    """
    df = pd.read_excel(input_file, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    # Validate required columns
    missing = [col for col in metrics_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns in Excel file: {missing}")

    # Keep subject column if available
    cols_to_export = (["subject"] + metrics_to_keep
                      if "subject" in df.columns else metrics_to_keep)
    df_selected = df[cols_to_export].copy()

    # Convert metrics to numeric
    for col in metrics_to_keep:
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

    # Compute summary statistics
    summary = pd.DataFrame(index=metrics_to_keep)
    summary["N"]    = df_selected[metrics_to_keep].count()
    summary["Mean"] = df_selected[metrics_to_keep].mean()
    summary["Std"]  = df_selected[metrics_to_keep].std(ddof=1)
    summary["Min"]  = df_selected[metrics_to_keep].min()
    summary["Max"]  = df_selected[metrics_to_keep].max()

    # Format Mean ± Std column
    def format_mean_std(metric_name, mean_val, std_val):
        if pd.isna(mean_val):
            return ""
        if pd.isna(std_val):
            std_val = 0
        if metric_name in percentage_metrics:
            return f"{mean_val * 100:.2f}% ± {std_val * 100:.2f}%"
        else:
            return f"{mean_val:.2f} ± {std_val:.2f}"

    summary["Mean ± Std"] = [
        format_mean_std(metric, summary.loc[metric, "Mean"], summary.loc[metric, "Std"])
        for metric in summary.index
    ]
    summary = summary.reset_index().rename(columns={"index": "Metric"})

    # Compute total counts for TP, FP, FN columns
    count_cols = [c for c in ["TP", "FP", "FN", "TP_CLU", "FN_CLU"]
                  if c in df_selected.columns]
    totals_df = None
    if count_cols:
        totals_df = pd.DataFrame({
            "Metric": count_cols,
            "Total":  [df_selected[c].sum() for c in count_cols]
        })

    return df_selected, summary, totals_df


def get_mean_std(summary_df, metric_name):
    """Extract mean and std for a given metric from a summary dataframe."""
    row = summary_df.loc[summary_df["Metric"] == metric_name]
    if row.empty:
        return None, None
    return row["Mean"].values[0], row["Std"].values[0]


def format_metric(metric_name, mean_val, std_val):
    """Format a metric value as 'mean ± std' (percentage) or 'mean' (counts)."""
    if mean_val is None or pd.isna(mean_val):
        return ""
    if std_val is None or pd.isna(std_val):
        std_val = 0

    if metric_name.replace("_CLU", "") in metrics_with_std:
        if metric_name in percentage_metrics:
            return f"{mean_val * 100:.2f}% ± {std_val * 100:.2f}%"
        else:
            return f"{mean_val:.2f} ± {std_val:.2f}"

    return f"{mean_val:.2f}"

# -----------------------------------------------------------------------------
# Process input files
# -----------------------------------------------------------------------------
df_full, summary_full, totals_full = process_file(
    input_file_full, metrics_full, label="FULL"
)

if COMPUTE_NO_CLU:
    df_no_clu, summary_no_clu, totals_no_clu = process_file(
        input_file_no_clu, metrics_no_clu, label="NO_CLU"
    )

# -----------------------------------------------------------------------------
# Build comparison table: rows = Normal / CLU / No CLU
#                         cols = Precision, Recall, F1, PQ, DSC, nDSC, TP, FP, FN
# -----------------------------------------------------------------------------
base_metrics = ["Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN"]
comparison_rows = []

# Normal row — standard metrics from full file
normal_row = {"Type": "Normal"}
for metric in base_metrics:
    mean_val, std_val = get_mean_std(summary_full, metric)
    normal_row[metric] = format_metric(metric, mean_val, std_val)
comparison_rows.append(normal_row)

# CLU row — CLU variants of Precision, Recall, F1, PQ from full file
clu_row = {"Type": "CLU"}
for metric in base_metrics:
    if metric in ["Precision", "Recall", "F1", "PQ"]:
        metric_name = f"{metric}_CLU"
        mean_val, std_val = get_mean_std(summary_full, metric_name)
        clu_row[metric] = format_metric(metric_name, mean_val, std_val)
    else:
        clu_row[metric] = ""
comparison_rows.append(clu_row)

# No CLU row — standard metrics from no-CLU file (if computed)
if COMPUTE_NO_CLU:
    no_clu_row = {"Type": "No CLU"}
    for metric in base_metrics:
        mean_val, std_val = get_mean_std(summary_no_clu, metric)
        no_clu_row[metric] = format_metric(metric, mean_val, std_val)
    comparison_rows.append(no_clu_row)

comparison_df = pd.DataFrame(comparison_rows)

# -----------------------------------------------------------------------------
# Save all sheets to Excel workbook
# -----------------------------------------------------------------------------
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_full.to_excel(writer,      sheet_name="lesion_selected_metrics", index=False)
    summary_full.to_excel(writer, sheet_name="lesion_summary_stats",    index=False)
    if totals_full is not None:
        totals_full.to_excel(writer, sheet_name="lesion_total_counts",  index=False)

    if COMPUTE_NO_CLU:
        df_no_clu.to_excel(writer,      sheet_name="lesion_no_clu_selected", index=False)
        summary_no_clu.to_excel(writer, sheet_name="lesion_no_clu_summary",  index=False)
        if totals_no_clu is not None:
            totals_no_clu.to_excel(writer, sheet_name="lesion_no_clu_totals", index=False)

    comparison_df.to_excel(writer, sheet_name="comparison_by_type", index=False)

print("Done.")
print(f"Excel output: {output_excel}")