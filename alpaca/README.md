# ALPaCA

This folder contains the scripts to run the ALPaCA (Automated Lesion, PRL, and CVS Analysis) pipeline. ALPaCA takes multi-modal MRI images as input and predicts whether lesion candidates are true lesions, Paramagnetic Rim Lesions (PRL) or Central Vein Sign (CVS) lesions.

For more details on the original ALPaCA project, refer to the [official ALPaCA repository](https://github.com/hufengling/ALPaCA).

---

## Files

- `make_predictions.R` — Corrected version of the original ALPaCA `make_predictions()` function (see Notes below)
- `run_alpaca.R` — Script to run the full ALPaCA pipeline on a list of subjects

---

## Requirements

The following R packages are required:
- `ALPaCA`
- `ANTsR`
- `ANTsRCore`

---

## Input folder structure

Each subject folder must contain the following files:

```
<input_root>/
└── <subject_id>/
    ├── T1.nii.gz       # T1-weighted MRI
    ├── FLAIR.nii.gz    # FLAIR MRI
    ├── EPIm.nii.gz     # EPI magnitude image
    └── EPIp.nii.gz     # EPI phase image
```

---

## Usage

```bash
Rscript run_alpaca.R <sublist_file> <input_root> <out_root> [chunk_size]
```

| Argument | Description |
|---|---|
| `sublist_file` | Path to a text file listing subject folder names, one per line |
| `input_root` | Path to the root directory containing subject folders |
| `out_root` | Path to the root directory where outputs will be saved |
| `chunk_size` | (Optional) Number of lesions to process per chunk — default is 50 |

**Example:**
```bash
Rscript run_alpaca.R subjects.txt /data/input /data/output
# With custom chunk size:
Rscript run_alpaca.R subjects.txt /data/input /data/output 100
```

---

## Outputs

For each subject, the following files are saved in `<out_root>/<subject_id>/`:

| File | Description |
|---|---|
| `alpaca_mask.nii.gz` | Final merged lesion mask |
| `predictions.csv` | Binary predictions (Lesion, PRL, CVS) |
| `probabilities.csv` | Raw predicted probabilities for each lesion |
| `prediction_uncertainties.csv` | Standard deviation across models and patches |

Preprocessed images (T1, FLAIR, EPI, phase) and intermediate files are also saved in the same folder.

---

## Notes on `make_predictions.R`

This file is a corrected version of the original `make_predictions()` function from the ALPaCA package. Two bugs were identified and fixed:

**Bug 1: Discordant predictions not reflected in output**

After `clear_discordant_predictions` cleans discordant cases in `lesion_sums` (e.g. PRL=1 or CVS=1 but Lesion=0), `binary_predictions` was never rebuilt from the cleaned `lesion_sums`. As a result, the corrections were correctly applied to `alpaca_mask` but silently ignored in the returned predictions dataframe and `predictions.csv`.

Fix — rebuild `binary_predictions` from `lesion_sums` after the cleanup step:
```r
binary_predictions[, 1] <- as.integer(lesion_sums %% 2 == 1)
binary_predictions[, 2] <- as.integer((lesion_sums %% 4) >= 2)
binary_predictions[, 3] <- as.integer(lesion_sums >= 4)
```

A pull request has been submitted to the ALPaCA authors. Until merged, use the corrected version provided here.