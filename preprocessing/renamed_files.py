import os
import shutil
import re

source_root = "/linux/luverheyen/data/images_raw/"
target_root = "/linux/luverheyen/data/dataset_renamed/"

os.makedirs(target_root, exist_ok=True)

# Expected files per subject
required_files = {"T1.nii.gz", "FLAIR.nii.gz", "EPIm.nii.gz", "EPIp.nii.gz"}

for file in os.listdir(source_root):
    if not file.endswith(".nii.gz"):
        continue

    # Extract subject ID (sub-XXX)
    match = re.search(r"(sub-\d+)", file)
    if not match:
        continue

    subject = match.group(1)
    subject_folder = os.path.join(target_root, subject)

    # ---- NEW: skip subject if already complete ----
    if os.path.exists(subject_folder):
        existing_files = set(os.listdir(subject_folder))
        if required_files.issubset(existing_files):
            print(f"Skipping {subject} (already complete)")
            continue

    os.makedirs(subject_folder, exist_ok=True)

    source_path = os.path.join(source_root, file)

    # Remove extension
    base = file[:-len(".nii.gz")]

    new_name = None

    if base.endswith("_FLAIR"):
        new_name = "FLAIR.nii.gz"

    elif base.endswith("_T1w"):
        new_name = "T1.nii.gz"

    elif base.endswith("phase_T2starw"):
        new_name = "EPIp.nii.gz"

    elif base.endswith("mag_T2starw"):
        new_name = "EPIm.nii.gz"

    else:
        continue

    target_path = os.path.join(subject_folder, new_name)

    if os.path.exists(target_path):
        print(f"WARNING: overwriting {subject}/{new_name} with {file}")

    shutil.copy2(source_path, target_path)
    print(f"Copied {file} -> {subject}/{new_name}")

print("Done.")