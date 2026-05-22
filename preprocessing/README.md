# Preprocessing

This folder contains scripts to prepare the raw MRI data before running any pipeline (ALPaCA, FLAMES, or APRL). These scripts should be run first before anything else in this repository.

---

## Files

- `rename_dataset.py` — Reorganizes and renames raw MRI files into the standardized folder structure required by all pipelines in this repository

---

## `rename_dataset.py`

The ALPaCA, FLAMES, and APRL pipelines all expect MRI files to be organized in a specific folder structure with standardized filenames. This script takes raw NIfTI files from a source directory, extracts the subject ID from each filename, and copies them into a per-subject folder with standardized names.

### Filename mapping

| Original suffix | Standardized name |
|---|---|
| `*_FLAIR.nii.gz` | `FLAIR.nii.gz` |
| `*_T1w.nii.gz` | `T1.nii.gz` |
| `*_phase_T2starw.nii.gz` | `EPIp.nii.gz` |
| `*_mag_T2starw.nii.gz` | `EPIm.nii.gz` |

### Usage

```bash
python rename_dataset.py <source_root> <target_root>
```

| Argument | Description |
|---|---|
| `source_root` | Path to the directory containing the raw NIfTI files |
| `target_root` | Path to the directory where the reorganized files will be saved |

**Example:**
```bash
python rename_dataset.py /data/images_raw /data/dataset_renamed
```

### Output structure

```
<target_root>/
└── <subject_id>/
    ├── T1.nii.gz
    ├── FLAIR.nii.gz
    ├── EPIm.nii.gz
    └── EPIp.nii.gz
```

### Notes

- Subjects whose folder already contains all four required files are skipped automatically. This means the script can be safely re-run if it is interrupted.
- Files with unrecognized suffixes are skipped with a warning message.
- The subject ID is extracted from the filename using the pattern `sub-XXX`. Files that do not contain this pattern are skipped with a warning.