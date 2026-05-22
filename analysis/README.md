# Analysis

This folder contains scripts to statistically analyze and visualize the segmentation metrics produced by the postprocessing pipeline. These scripts should be run after the postprocessing step.

---

## Files

- `box_plot.py` — Boxplots comparing segmentation metrics between two models or for a single model
- `bland_altman_plot.py` — Bland-Altman plots assessing agreement between predicted and reference lesion counts
- `roc_plot.py` — ROC curves with AUROC and optimal threshold for lesion and PRL classification
- `shapiro_wilk.py` — Shapiro-Wilk normality test on segmentation metrics across all method/target combinations

---

## Requirements

**Python packages:**
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `openpyxl`
- `pathlib`

---

## `box_plot.py`

Generates boxplots comparing segmentation metrics (PQ, DSC, nDSC, F1, Recall, Precision, and CLU variants) between two models (ALPaCA vs APRL or MIMoSA vs FLaMeS), or for a single model alone. Each metric is shown as a separate panel with jittered subject-level points, mean diamond markers, and Wilcoxon/Mann-Whitney p-value annotations when comparing two models.

Input files are the processed Excel workbooks produced by `summarize_metrics.py` (postprocessing step).

### Usage

```bash
python box_plot.py --target <target> --comp <comp>
                   [--prl_suffix <prl_suffix>]
                   [--single_model <single_model>]
                   [--input_dir <input_dir>]
                   [--output_dir <output_dir>]
```

| Argument | Description |
|---|---|
| `--target` | Target type: `lesion` or `prl` (required) |
| `--comp` | Comparison model: `flames` or `aprl` (required) |
| `--prl_suffix` | PRL suffix in folder names: `""`, `_PRL_pred`, or `_PRL_ref` (default: `_PRL_ref`) |
| `--single_model` | Plot a single model only: `alpaca`, `flames`, or `aprl` (default: plots both) |
| `--input_dir` | Root directory containing the `output_comp/` folder (default: current directory) |
| `--output_dir` | Directory where output figures will be saved (default: `boxplot_outputs/`) |

**Example:**
```bash
# Compare ALPaCA vs APRL for PRL target
python box_plot.py --target prl --comp aprl --prl_suffix _PRL_ref

# Single model only
python box_plot.py --target prl --comp aprl --single_model alpaca

# Custom directories
python box_plot.py --target prl --comp aprl \
        --input_dir /data/results --output_dir /data/figures
```

**Output:** One PDF figure saved in `--output_dir`.

---

## `bland_altman_plot.py`

Generates Bland-Altman plots for two models (A and B) to assess agreement between predicted and reference lesion counts. Each figure contains a lesion count row and optionally a CLU count row. Displays mean difference, limits of agreement (±1.96 SD), and a proportional bias test (R², p-value).

### Usage

```bash
python bland_altman_plot.py --input_a <input_a> --input_b <input_b>
                             [--label_a <label_a>] [--label_b <label_b>]
                             [--output_a <output_a>] [--output_b <output_b>]
                             [--include_clu]
                             [--xlim_lesion MIN MAX] [--ylim_lesion MIN MAX]
                             [--xlim_clu MIN MAX] [--ylim_clu MIN MAX]
```

| Argument | Description |
|---|---|
| `--input_a` | Excel metrics file for model A (required) |
| `--input_b` | Excel metrics file for model B (required) |
| `--label_a` | Display name for model A (default: `Model A`) |
| `--label_b` | Display name for model B (default: `Model B`) |
| `--output_a` | Output image for model A (default: `bland_altman_a.png`) |
| `--output_b` | Output image for model B (default: `bland_altman_b.png`) |
| `--include_clu` | Add a CLU Count row to each figure (default: False) |
| `--xlim_lesion` | X-axis limits for lesion row: two values `MIN MAX` (default: auto) |
| `--ylim_lesion` | Y-axis limits for lesion row: two values `MIN MAX` (default: auto) |
| `--xlim_clu` | X-axis limits for CLU row: two values `MIN MAX` (default: auto) |
| `--ylim_clu` | Y-axis limits for CLU row: two values `MIN MAX` (default: auto) |

**Example:**
```bash
python bland_altman_plot.py \
    --input_a metrics_alpaca.xlsx --label_a ALPaCA \
    --input_b metrics_flames.xlsx --label_b FLaMeS \
    --include_clu --xlim_lesion -10 110 --ylim_lesion -50 130
```

**Output:** Two PNG figures, one per model.

---

## `roc_plot.py`

Generates ROC curve figures for lesion and PRL classification from ALPaCA predictions. For each task, the script computes the AUROC with a 95% bootstrap confidence interval (2000 resamples) and identifies the optimal threshold using Youden's J statistic.

Input Excel file must contain two sheets: `lesion` (columns `true_label`, `Lesion`) and `prl` (columns `true_label`, `PRL`).

### Usage

```bash
python roc_plot.py --input <input>
                   [--output_lesion <output_lesion>]
                   [--output_prl <output_prl>]
```

| Argument | Description |
|---|---|
| `--input` | Path to the input Excel file containing ROC data (required) |
| `--output_lesion` | Output path for the lesion ROC figure (default: `roc_lesion.pdf`) |
| `--output_prl` | Output path for the PRL ROC figure (default: `roc_prl.pdf`) |

**Example:**
```bash
python roc_plot.py --input /data/results/roc_data.xlsx
python roc_plot.py --input /data/results/roc_data.xlsx \
        --output_lesion /data/figures/roc_lesion.pdf \
        --output_prl /data/figures/roc_prl.pdf
```

**Output:** Two PDF figures, one for lesion and one for PRL classification.

---

## `shapiro_wilk.py`

Runs the Shapiro-Wilk normality test on segmentation metrics across all method/target/PRL-suffix combinations (ALPaCA, APRL, FLaMeS × lesion, prl × no suffix, `_PRL_pred`, `_PRL_ref`). Input files are the processed Excel workbooks produced by `summarize_metrics.py`. Files that do not exist are skipped with a warning, so the script can be run even if only a subset of combinations have been computed.

### Usage

```bash
python shapiro_wilk.py [--input_dir <input_dir>]
                       [--output <output>]
```

| Argument | Description |
|---|---|
| `--input_dir` | Root directory containing the `output_comp/` folder (default: current directory) |
| `--output` | Output Excel file path (default: `normality_check.xlsx`) |

**Example:**
```bash
python shapiro_wilk.py --input_dir /data/results
python shapiro_wilk.py --input_dir /data/results --output /data/results/normality.xlsx
```

**Output:** An Excel file with one row per (method, metric) combination containing the W statistic, p-value, and a `Normal?` column (`Yes` if p > 0.05), plus a printed summary of how many tests passed normality.