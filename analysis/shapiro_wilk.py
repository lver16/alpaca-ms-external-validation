import pandas as pd
from pathlib import Path
from scipy import stats

# =========================================================
# SETTINGS
# =========================================================

metrics_all = ["PQ", "DSC", "nDSC", "F1", "Recall", "Precision",
               "F1_CLU", "Recall_CLU", "Precision_CLU"]

sheet_name = "lesion_selected_metrics"

files = {
    "ALPaCA_lesion" : Path("output_comp/alpaca_metrics_with_wo_CLU/processed/metrics_summary_lesion_comparison.xlsx"),
    "APRL_lesion"   : Path("output_comp/aprl_metrics_with_wo_CLU/processed/metrics_summary_lesion_comparison.xlsx"),
    "FLAMeS_lesion" : Path("output_comp/alpaca_metrics_with_wo_CLU_flames_05_thresh/processed/metrics_summary_lesion_comparison.xlsx"),
    "ALPaCA_prl"    : Path("output_comp/alpaca_metrics_with_wo_CLU/processed/metrics_summary_prl_comparison.xlsx"),
    "APRL_prl"      : Path("output_comp/aprl_metrics_with_wo_CLU/processed/metrics_summary_prl_comparison.xlsx"),
    "FLAMeS_prl"    : Path("output_comp/alpaca_metrics_with_wo_CLU_flames_05_thresh/processed/metrics_summary_prl_comparison.xlsx"),
    "ALPaCA_prl_pred":Path("output_comp/alpaca_metrics_with_wo_CLU_PRL_pred/processed/metrics_summary_prl_comparison.xlsx"),
    "APRL_prl_pred"  :Path("output_comp/aprl_metrics_with_wo_CLU_PRL_pred/processed/metrics_summary_prl_comparison.xlsx"),
    "FLAMeS_prl_pred":Path("output_comp/alpaca_metrics_with_wo_CLU_flames_05_thresh_PRL_pred/processed/metrics_summary_prl_comparison.xlsx"),
    "FLAMeS_prl_ref" :Path("output_comp/alpaca_metrics_with_wo_CLU_flames_05_thresh_PRL_ref/processed/metrics_summary_prl_comparison.xlsx"),
    "ALPaCA_prl_ref":Path("output_comp/alpaca_metrics_with_wo_CLU_PRL_ref/processed/metrics_summary_prl_comparison.xlsx"),
    "APRL_prl_ref"  :Path("output_comp/aprl_metrics_with_wo_CLU_PRL_ref/processed/metrics_summary_prl_comparison.xlsx"),
    "APRL_lesion_05":Path("output_comp/aprl_metrics_with_wo_CLU_50/processed/metrics_summary_lesion_comparison.xlsx"),
    "APRL_prl_05"    :Path("output_comp/aprl_metrics_with_wo_CLU_50/processed/metrics_summary_prl_comparison.xlsx"),
}

# =========================================================
# RUN SHAPIRO-WILK
# =========================================================

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
            print(f"[WARNING] Not enough values for '{metric}' in {label}, skipping.")
            continue

        stat, p = stats.shapiro(vals)

        rows.append({
            "Method"    : label,
            "Metric"    : metric,
            "N"         : len(vals),
            "W-stat"    : round(stat, 4),
            "p-value"   : round(p, 4),
            "Normal?"   : "Yes" if p > 0.05 else "No"
        })

# =========================================================
# OUTPUT
# =========================================================

df_result = pd.DataFrame(rows)

# Print full table
print("\n===== SHAPIRO-WILK NORMALITY TEST =====\n")
print(df_result.to_string(index=False))

# Summary
n_total  = len(df_result)
n_normal = (df_result["Normal?"] == "Yes").sum()
n_not    = (df_result["Normal?"] == "No").sum()
print(f"\n--- Summary ---")
print(f"Total tests   : {n_total}")
print(f"Normal        : {n_normal} ({100*n_normal/n_total:.1f}%)")
print(f"Non-normal    : {n_not} ({100*n_not/n_total:.1f}%)")

# Save to Excel with conditional formatting hint
output_path = Path("normality_check.xlsx")
df_result.to_excel(output_path, index=False)
print(f"\nSaved to: {output_path}")