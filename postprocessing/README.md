# Postprocessing

This folder contains scripts to postprocess and evaluate the outputs of the ALPaCA and APRL pipelines. The scripts compute segmentation metrics against reference masks, fix known issues in ALPaCA output files, and summarize results into Excel workbooks.

---

## Folder structure

```
postprocessing/
├── fix_predictions.py          # Fix ALPaCA output files before computing metrics
├── summarize_metrics.py        # Summarize metrics into a comparison Excel workbook
├── alpaca/
│   └── compute_alpaca_metrics.py   # Compute lesion and PRL metrics for ALPaCA
└── aprl/
    └── compute_aprl_metrics.py     # Compute lesion and PRL metrics for APRL
```

---

## Recommended workflow

```
1. fix_predictions.py           — fix ALPaCA output files (ALPaCA only)
2. compute_alpaca_metrics.py    — compute metrics for ALPaCA predictions
   or compute_aprl_metrics.py   — compute metrics for APRL predictions
3. summarize_metrics.py         — summarize and compare metrics across configurations
```

---

## Requirements

**Python packages:**
- `pandas`
- `numpy`
- `nibabel`
- `openpyxl`
- `pathlib`

**External dependency:**
- `conflunet` — required by `compute_alpaca_metrics.py` and `compute_aprl_metrics.py` for instance matching and confluent lesion detection. Pass the path to your local conflunet installation via `--conflunet_dir`.

---

## `fix_predictions.py`

Fixes known issues in ALPaCA `predictions.csv` and `probabilities.csv` files before running metric computation. Two issues are addressed:

- **Missing index column** — some versions of ALPaCA save CSV files without a row index, causing misalignment when reading them back
- **Discordant predictions** — candidates where PRL=1 or CVS=1 but Lesion=0, which should be zeroed out since lesion prediction is more reliable

Original files are archived in each subject's `archive/` subfolder before modification.

**Note:** Run this before `compute_alpaca_metrics.py` to ensure correct results. Use `--dry_run` to preview changes without modifying any file.

### Usage

```bash
python fix_predictions.py --alpaca_dir <alpaca_dir> [--dry_run]
```

| Argument | Description |
|---|---|
| `--alpaca_dir` | Root directory containing one subfolder per subject (ALPaCA output) |
| `--dry_run` | Preview changes without modifying any file |

**Example:**
```bash
python fix_predictions.py --alpaca_dir /data/alpaca_out
python fix_predictions.py --alpaca_dir /data/alpaca_out --dry_run
```

---

## `alpaca/compute_alpaca_metrics.py`

Computes lesion and PRL segmentation metrics for ALPaCA predictions against reference instance label masks. Evaluations are run for:
- **Lesion** (full): all candidates with V1=1, V2=1, or V3=1
- **PRL** (full): candidates with V2=1
- **Lesion no-CLU** (optional): same as lesion but confluent lesions removed from reference
- **PRL no-CLU** (optional): same as PRL but confluent lesions removed from reference

### Usage

```bash
python compute_alpaca_metrics.py \
    --alpaca_dir <alpaca_dir> \
    --ref_dir <ref_dir> \
    --out_dir <out_dir> \
    --conflunet_dir <conflunet_dir> \
    [--prl_filter <prl_filter>] \
    [--compute_no_clu]
```

| Argument | Description |
|---|---|
| `--alpaca_dir` | Path to ALPaCA output directory (one subfolder per subject) |
| `--ref_dir` | Path to directory containing reference instance label masks |
| `--out_dir` | Path to directory where output Excel files will be saved |
| `--conflunet_dir` | Path to the conflunet package directory |
| `--prl_filter` | Subject filter: `all`, `ref`, `pred`, or `ref_or_pred` (default: `ref`) |
| `--compute_no_clu` | Also compute metrics after removing confluent lesions (default: False) |

**Example:**
```bash
python compute_alpaca_metrics.py \
    --alpaca_dir /data/alpaca_out \
    --ref_dir /data/labels_raw \
    --out_dir /data/alpaca_metrics \
    --conflunet_dir /conflunet/evaluation \
    --prl_filter ref --compute_no_clu
```

### Outputs

| File | Description |
|---|---|
| `metrics_lesion.xlsx` | Lesion metrics per subject (full) |
| `metrics_prl.xlsx` | PRL metrics per subject (full) |
| `metrics_lesion_no_clu.xlsx` | Lesion metrics without confluent lesions (if `--compute_no_clu`) |
| `metrics_prl_no_clu.xlsx` | PRL metrics without confluent lesions (if `--compute_no_clu`) |

---

## `aprl/compute_aprl_metrics.py`

Computes lesion and PRL segmentation metrics for APRL predictions against reference instance label masks. Similar to the ALPaCA version but adapted for APRL output format (`aprl_leslabels.nii.gz` and `aprl_preds.csv`). No-CLU evaluation is always computed for APRL.

Key difference from ALPaCA: all candidates in the APRL CSV are considered lesion detections. A candidate is classified as PRL if `rimpos > prl_threshold`.

### Usage

```bash
python compute_aprl_metrics.py \
    --aprl_dir <aprl_dir> \
    --ref_dir <ref_dir> \
    --out_dir <out_dir> \
    --conflunet_dir <conflunet_dir> \
    [--prl_filter <prl_filter>] \
    [--prl_threshold <prl_threshold>]
```

| Argument | Description |
|---|---|
| `--aprl_dir` | Path to APRL output directory (one subfolder per subject) |
| `--ref_dir` | Path to directory containing reference instance label masks |
| `--out_dir` | Path to directory where output Excel files will be saved |
| `--conflunet_dir` | Path to the conflunet package directory |
| `--prl_filter` | Subject filter: `all`, `ref`, `pred`, or `ref_or_pred` (default: `all`) |
| `--prl_threshold` | Probability threshold for PRL classification (default: `0.5`) |

**Example:**
```bash
python compute_aprl_metrics.py \
    --aprl_dir /data/processed_50_custom \
    --ref_dir /data/labels_raw \
    --out_dir /data/aprl_metrics \
    --conflunet_dir /conflunet/evaluation \
    --prl_filter ref --prl_threshold 0.6
```

### Outputs

| File | Description |
|---|---|
| `metrics_lesion.xlsx` | Lesion metrics per subject (full) |
| `metrics_lesion_no_clu.xlsx` | Lesion metrics without confluent lesions |
| `metrics_prl.xlsx` | PRL metrics per subject (full) |
| `metrics_prl_no_clu.xlsx` | PRL metrics without confluent lesions |

---

## `summarize_metrics.py`

Processes the Excel metric files produced by `compute_alpaca_metrics.py` or `compute_aprl_metrics.py` and generates a summary workbook containing per-subject metrics, summary statistics (mean, std, min, max), total counts, and a comparison table across Normal, CLU, and optionally No-CLU configurations.

### Usage

```bash
python summarize_metrics.py \
    --target <target> \
    --method_prl <method_prl> \
    [--segmentation <segmentation>] \
    [--prl_suffix <prl_suffix>] \
    [--compute_no_clu] \
    [--input_dir <input_dir>]
```

| Argument | Description |
|---|---|
| `--target` | Target type: `lesion` or `prl` |
| `--method_prl` | Method used: `aprl` or `alpaca` |
| `--segmentation` | Segmentation suffix in folder name: `""`, `_flames`, or `_flames_05_thresh` (default: `""`) |
| `--prl_suffix` | PRL suffix in folder name: `_50_PRL_ref`, `""`, or `_PRL_pred` (default: `_50_PRL_ref`) |
| `--compute_no_clu` | Also process the no-CLU metrics file (default: False) |
| `--input_dir` | Root directory containing the `output_comp/` folder (default: current directory) |

**Example:**
```bash
python summarize_metrics.py --target prl --method_prl aprl
python summarize_metrics.py --target lesion --method_prl alpaca \
        --segmentation _flames --prl_suffix _PRL_pred --compute_no_clu
```

### Outputs

Saved in `<input_dir>/output_comp/<method_prl>_metrics_with_wo_CLU<segmentation><prl_suffix>/processed/`:

| Sheet | Description |
|---|---|
| `lesion_selected_metrics` | Per-subject metrics |
| `lesion_summary_stats` | Mean, std, min, max per metric |
| `lesion_total_counts` | Total TP, FP, FN counts |
| `comparison_by_type` | Comparison table across Normal, CLU, and No-CLU |