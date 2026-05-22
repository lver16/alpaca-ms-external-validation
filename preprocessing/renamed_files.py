"""
rename_dataset.py
-------------------------------------------------------------------------------
Description:
    Reorganizes and renames raw MRI files into a standardized folder structure
    required by the ALPaCA and APRL pipelines. The script scans a source
    directory for NIfTI files, extracts the subject ID from each filename,
    creates a folder per subject in the target directory, and copies each file
    with a standardized name.

    The following filename suffixes are recognized and renamed:
        *_FLAIR.nii.gz          -> FLAIR.nii.gz
        *_T1w.nii.gz            -> T1.nii.gz
        *_phase_T2starw.nii.gz  -> EPIp.nii.gz
        *_mag_T2starw.nii.gz    -> EPIm.nii.gz

    Subjects whose folder already contains all four required files are skipped
    automatically.

Usage:
    python rename_dataset.py <source_root> <target_root>

Arguments:
    source_root   Path to the directory containing the raw NIfTI files
    target_root   Path to the directory where the reorganized files will be saved

Example:
    python rename_dataset.py /data/images_raw /data/dataset_renamed

Output:
    For each subject found in source_root, creates a folder:
        <target_root>/<subject_id>/
            ├── T1.nii.gz
            ├── FLAIR.nii.gz
            ├── EPIm.nii.gz
            └── EPIp.nii.gz
"""

import argparse
import os
import re
import shutil

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Rename and reorganize raw MRI files into a standardized folder structure."
)
parser.add_argument("source_root", help="Directory containing the raw NIfTI files")
parser.add_argument("target_root", help="Directory where reorganized files will be saved")
args = parser.parse_args()

source_root = args.source_root
target_root = args.target_root

if not os.path.exists(source_root):
    raise FileNotFoundError(f"Source directory does not exist: {source_root}")

os.makedirs(target_root, exist_ok=True)

# -----------------------------------------------------------------------------
# Define required files per subject
# -----------------------------------------------------------------------------
required_files = {"T1.nii.gz", "FLAIR.nii.gz", "EPIm.nii.gz", "EPIp.nii.gz"}

# -----------------------------------------------------------------------------
# Process each NIfTI file in the source directory
# -----------------------------------------------------------------------------
for file in sorted(os.listdir(source_root)):
    if not file.endswith(".nii.gz"):
        continue

    # Extract subject ID (sub-XXX) from filename
    match = re.search(r"(sub-\d+)", file)
    if not match:
        print(f"WARNING: could not extract subject ID from {file}, skipping.")
        continue

    subject = match.group(1)
    subject_folder = os.path.join(target_root, subject)

    # Skip subject if all required files are already present
    if os.path.exists(subject_folder):
        existing_files = set(os.listdir(subject_folder))
        if required_files.issubset(existing_files):
            print(f"Skipping {subject} (already complete)")
            continue

    os.makedirs(subject_folder, exist_ok=True)

    # Determine standardized output filename based on input filename suffix
    base = file[:-len(".nii.gz")]

    if base.endswith("_FLAIR"):
        new_name = "FLAIR.nii.gz"
    elif base.endswith("_T1w"):
        new_name = "T1.nii.gz"
    elif base.endswith("phase_T2starw"):
        new_name = "EPIp.nii.gz"
    elif base.endswith("mag_T2starw"):
        new_name = "EPIm.nii.gz"
    else:
        print(f"WARNING: unrecognized file suffix for {file}, skipping.")
        continue

    # Copy file to target folder with standardized name
    source_path = os.path.join(source_root, file)
    target_path = os.path.join(subject_folder, new_name)

    if os.path.exists(target_path):
        print(f"WARNING: overwriting {subject}/{new_name} with {file}")

    shutil.copy2(source_path, target_path)
    print(f"Copied {file} -> {subject}/{new_name}")

print("\nDone.")