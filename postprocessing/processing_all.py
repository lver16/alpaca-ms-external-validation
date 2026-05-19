import pandas as pd
from pathlib import Path

# =========================================================
# SETTINGS
# =========================================================
target = "prl" # lesion or prl

segmentation = "" # "" or _flames or _flames_05_thresh

method_prl = "aprl" # aprl or alpaca

PRL = "_50_PRL_ref"  # "" or "_PRL_pred" or "_PRL_ref"

COMPUTE_NO_CLU = False   # ← set to False if no-CLU files were not computed

input_file_full = Path(f"output_comp/{method_prl}_metrics_with_wo_CLU{segmentation}{PRL}/metrics_{target}.xlsx")
if COMPUTE_NO_CLU:
    input_file_no_clu = Path(f"output_comp/{method_prl}_metrics_with_wo_CLU{segmentation}{PRL}/metrics_{target}_no_clu.xlsx")

output_dir = Path(f"output_comp/{method_prl}_metrics_with_wo_CLU{segmentation}{PRL}/processed")
output_dir.mkdir(exist_ok=True)

output_excel = output_dir / f"metrics_summary_{target}_comparison.xlsx"

# Metrics for FULL lesion file
metrics_full = [
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN",
    "Precision_CLU", "Recall_CLU", "F1_CLU", "PQ_CLU"
]

# Metrics for NO_CLU lesion file
metrics_no_clu = [
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN"
]

# Metrics shown as percentages
percentage_metrics = {
    "Precision", "Recall", "F1", "PQ", "DSC", "nDSC",
    "Precision_CLU", "Recall_CLU", "F1_CLU", "PQ_CLU"
}

# Metrics that should be shown with std in comparison sheet
metrics_with_std = {"Precision", "Recall", "F1", "PQ", "DSC", "nDSC"}

# =========================================================
# HELPER FUNCTION
# =========================================================
def process_file(input_file: Path, metrics_to_keep: list, label: str):
    df = pd.read_excel(input_file, sheet_name=0)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Check required columns
    missing = [col for col in metrics_to_keep if col not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Missing required columns in Excel file: {missing}")

    # Keep subject if available
    cols_to_export = ["subject"] + metrics_to_keep if "subject" in df.columns else metrics_to_keep
    df_selected = df[cols_to_export].copy()

    # Convert metrics to numeric
    for col in metrics_to_keep:
        df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

    # Summary statistics
    summary = pd.DataFrame(index=metrics_to_keep)
    summary["N"] = df_selected[metrics_to_keep].count()
    summary["Mean"] = df_selected[metrics_to_keep].mean()
    summary["Std"] = df_selected[metrics_to_keep].std(ddof=1)
    summary["Min"] = df_selected[metrics_to_keep].min()
    summary["Max"] = df_selected[metrics_to_keep].max()

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

    # Totals
    count_cols = [c for c in ["TP", "FP", "FN", "TP_CLU", "FN_CLU"] if c in df_selected.columns]
    totals_df = None
    if count_cols:
        totals_df = pd.DataFrame({
            "Metric": count_cols,
            "Total": [df_selected[c].sum() for c in count_cols]
        })

    return df_selected, summary, totals_df


# =========================================================
# PROCESS BOTH FILES
# =========================================================
df_full, summary_full, totals_full = process_file(
    input_file_full,
    metrics_full,
    label="FULL"
)

if COMPUTE_NO_CLU:
    df_no_clu, summary_no_clu, totals_no_clu = process_file(
        input_file_no_clu,
        metrics_no_clu,
        label="NO_CLU"
    )

# =========================================================
# BUILD COMPARISON SHEET
# rows = Normal / CLU / No CLU
# cols = Precision, Recall, F1, PQ, TP, FP, FN
# percentage metrics -> mean ± std
# count metrics -> mean only
# =========================================================
base_metrics = ["Precision", "Recall", "F1", "PQ", "DSC", "nDSC", "TP", "FP", "FN"]

def get_mean_std(summary_df, metric_name):
    row = summary_df.loc[summary_df["Metric"] == metric_name]
    if row.empty:
        return None, None
    mean_val = row["Mean"].values[0]
    std_val = row["Std"].values[0]
    return mean_val, std_val

def format_metric(metric_name, mean_val, std_val):
    if mean_val is None or pd.isna(mean_val):
        return ""
    if std_val is None or pd.isna(std_val):
        std_val = 0

    # Metrics with std
    if metric_name.replace("_CLU", "") in metrics_with_std:
        if metric_name in percentage_metrics:
            return f"{mean_val * 100:.2f}% ± {std_val * 100:.2f}%"
        else:
            return f"{mean_val:.2f} ± {std_val:.2f}"

    # Count metrics without std
    return f"{mean_val:.2f}"

comparison_rows = []

# Normal
normal_row = {"Type": "Normal"}
for metric in base_metrics:
    mean_val, std_val = get_mean_std(summary_full, metric)
    normal_row[metric] = format_metric(metric, mean_val, std_val)
comparison_rows.append(normal_row)

# CLU
clu_row = {"Type": "CLU"}
for metric in base_metrics:
    if metric in ["Precision", "Recall", "F1", "PQ"]:
        metric_name = f"{metric}_CLU"
        mean_val, std_val = get_mean_std(summary_full, metric_name)
        clu_row[metric] = format_metric(metric_name, mean_val, std_val)
    else:
        clu_row[metric] = ""
comparison_rows.append(clu_row)

# No CLU
if COMPUTE_NO_CLU:
    no_clu_row = {"Type": "No CLU"}
    for metric in base_metrics:
        mean_val, std_val = get_mean_std(summary_no_clu, metric)
        no_clu_row[metric] = format_metric(metric, mean_val, std_val)
    comparison_rows.append(no_clu_row)

comparison_df = pd.DataFrame(comparison_rows)

# =========================================================
# SAVE
# =========================================================
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_full.to_excel(writer, sheet_name="lesion_selected_metrics", index=False)
    summary_full.to_excel(writer, sheet_name="lesion_summary_stats", index=False)
    if totals_full is not None:
        totals_full.to_excel(writer, sheet_name="lesion_total_counts", index=False)

    if COMPUTE_NO_CLU:
        df_no_clu.to_excel(writer, sheet_name="lesion_no_clu_selected", index=False)
        summary_no_clu.to_excel(writer, sheet_name="lesion_no_clu_summary", index=False)
        if totals_no_clu is not None:
            totals_no_clu.to_excel(writer, sheet_name="lesion_no_clu_totals", index=False)

    comparison_df.to_excel(writer, sheet_name="comparison_by_type", index=False)

print("Done.")
print(f"Excel output: {output_excel}")