# FLAMES + ALPaCA

This folder contains the scripts to run the ALPaCA pipeline using FLAMES (Fast Lesion and Multiple Sclerosis Segmentation) probability maps as lesion candidates, bypassing the MIMoSA step used in the original ALPaCA pipeline.

For more details on the original FLAMES project, refer to the [official FLAMES repository](https://github.com/hufnagel-lab/FLAMES).

---

## Workflow

The three scripts must be run in the following order:

```
1. (Optional) undo_brainmask.py       — restore FLAMES maps to original image space
2. preprocess_images_flames.R         — preprocess MRI images using FLAMES map
3. run_flames_alpaca.R                — run full FLAMES + ALPaCA pipeline
```

Step 1 is optional and depends on your dataset — see details below.

---

## Files

- `undo_brainmask.py` — Restores FLAMES probability maps to the original image space after brain masking
- `preprocess_images_flames.R` — Modified version of ALPaCA's `preprocess_images()` that uses FLAMES maps instead of MIMoSA
- `run_flames_alpaca.R` — Script to run the full FLAMES + ALPaCA pipeline on a list of subjects

---

## Requirements

**R packages:**
- `ALPaCA`
- `ANTsR`
- `ANTsRCore`
- `extrantsr`
- `oro.nifti`
- `neurobase`
- `fslr`
- `WhiteStripe`

**Python packages:**
- `nibabel`
- `numpy`

---

## Step 1 (Optional): Undo brain masking — `undo_brainmask.py`

This step is only needed if your FLAMES probability maps were generated from brain-masked images. When FLAMES is run on brain-masked images, its output is cropped to the bounding box of the brain mask and no longer matches the original image space. Since ALPaCA requires all input images to share the same image space, this script restores the FLAMES probability maps to the full original image space before passing them to ALPaCA.

**Skip this step if FLAMES was run on full images without prior brain masking.**

### Usage

```bash
python undo_brainmask.py <flames_dir> <brainmask_dir> <output_dir>
```

| Argument | Description |
|---|---|
| `flames_dir` | Directory containing FLAMES probability maps (`sub-*_pred-prob.nii.gz`) |
| `brainmask_dir` | Directory containing brain masks (`<subject_id>_brainmask.nii.gz`) |
| `output_dir` | Directory where restored images will be saved (can be the same as `flames_dir` to overwrite in place) |

**Example:**
```bash
python undo_brainmask.py /data/flames_acls /data/brainmasks /data/flames_restored
```

---

## Step 2: Preprocess images — `preprocess_images_flames.R`

This is a modified version of the `preprocess_images()` function from the ALPaCA package. The original function uses MIMoSA to generate lesion candidates from T1 and FLAIR images. This version bypasses MIMoSA and instead uses a pre-computed FLAMES probability map to generate lesion candidates directly.

All other preprocessing steps are identical to the original ALPaCA pipeline: bias correction, registration of T1 and FLAIR to EPI space, brain masking, and intensity normalization.

This function is called internally by `run_flames_alpaca.R` and does not need to be run separately.

---

## Step 3: Run FLAMES + ALPaCA — `run_flames_alpaca.R`

This script runs the full FLAMES + ALPaCA pipeline on a list of subjects. It calls `preprocess_images_flames()` for preprocessing and then runs ALPaCA predictions in chunks to manage memory.

### Input folder structure

Each subject folder must contain the following files:

```
<input_root>/
└── <subject_id>/
    ├── T1.nii.gz       # T1-weighted MRI
    ├── FLAIR.nii.gz    # FLAIR MRI
    ├── EPIm.nii.gz     # EPI magnitude image
    └── EPIp.nii.gz     # EPI phase image
```

Additional input files required per subject:
```
<flames_dir>/<subject_id>_ses-01_pred-prob.nii.gz   # FLAMES probability map
<brainmask_dir>/<subject_id>_brainmask.nii.gz        # Brain mask
```

### Usage

```bash
Rscript run_flames_alpaca.R <sublist_file> <input_root> <out_root> \
        <flames_root> <brainmask_root> <alpaca_src> \
        [flames_threshold] [chunk_size]
```

| Argument | Description |
|---|---|
| `sublist_file` | Path to a text file listing subject IDs, one per line |
| `input_root` | Path to the root directory containing subject folders |
| `out_root` | Path to the root directory where outputs will be saved |
| `flames_root` | Path to the directory containing FLAMES probability maps |
| `brainmask_root` | Path to the directory containing brain masks |
| `alpaca_src` | Path to the ALPaCA source directory (`ALPaCA/R/`) |
| `flames_threshold` | (Optional) Probability threshold for FLAMES map binarization — default is 0.5 |
| `chunk_size` | (Optional) Number of lesions to process per chunk — default is 50 |

**Example:**
```bash
Rscript run_flames_alpaca.R subjects.txt /data/input /data/output \
        /data/flames /data/brainmasks /path/to/ALPaCA/R
# With custom threshold and chunk size:
Rscript run_flames_alpaca.R subjects.txt /data/input /data/output \
        /data/flames /data/brainmasks /path/to/ALPaCA/R 0.3 100
```

### Outputs

For each subject, the following files are saved in `<out_root>/<subject_id>/`:

| File | Description |
|---|---|
| `alpaca_mask.nii.gz` | Final merged lesion mask |
| `predictions.csv` | Binary predictions (Lesion, PRL, CVS) |
| `probabilities.csv` | Raw predicted probabilities for each lesion |
| `prediction_uncertainties.csv` | Standard deviation across models and patches |
| `prob.nii.gz` | FLAMES probability map resampled to EPI space |
| `labeled_candidates.nii.gz` | Labeled lesion candidates derived from FLAMES map |
| `eroded_candidates.nii.gz` | Eroded lesion candidates |

Preprocessed images (T1, FLAIR, EPI, phase) are also saved in the same folder.