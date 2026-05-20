"""
undo_brainmask.py
-------------------------------------------------------------------------------
Description:
    FLAMES generates a lesion probability map from MRI images. When FLAMES is
    run on brain-masked images, its output probability map is cropped to the
    bounding box of the brain mask and no longer matches the original image
    space. However, ALPaCA requires all input images to share the same image
    space. This script restores the FLAMES probability maps to the original
    image space by placing them back into a full-size volume using the brain
    mask bounding box, making them compatible with the ALPaCA pipeline.

    NOTE: This step is optional and depends on your dataset. Only run this
    script if your MRI images were brain-masked before running FLAMES. If
    FLAMES was run on full images without prior brain masking, skip this step.

Usage:
    python undo_brainmask.py <flames_dir> <brainmask_dir> <output_dir>

Arguments:
    flames_dir      Path to the directory containing FLAMES probability maps
                    Files must match the pattern: sub-*_pred-prob.nii.gz
    brainmask_dir   Path to the directory containing brain masks
                    Files must match the pattern: <subject_id>_brainmask.nii.gz
    output_dir      Path to the directory where restored images will be saved
                    (can be the same as flames_dir to overwrite in place)

Example:
    python undo_brainmask.py /data/flames_acls /data/brainmasks /data/flames_restored

Output:
    For each subject, saves a restored probability map in float32 format
    with the same filename as the input, using the affine of the brain mask.
"""

import argparse
import nibabel as nib
import numpy as np
import os
from os.path import join as pjoin


def undo_brainmask(brainmask, instance_mask):
    """
    Undo brain masking on an instance mask.

    Parameters:
    - brainmask (numpy.ndarray): A binary brain mask where non-brain regions
      are 0, sized [H_orig, W_orig, D_orig].
    - instance_mask (numpy.ndarray): An instance segmentation mask that has
      been brain-masked, sized [H_masked, W_masked, D_masked].

    Returns:
    - numpy.ndarray: The instance segmentation mask with non-brain regions
      restored, sized [H_orig, W_orig, D_orig].
    """
    # Find bounding box of the brain mask
    brainmask_indices = np.where(brainmask > 0)
    x_min, x_max = brainmask_indices[0].min(), brainmask_indices[0].max()
    y_min, y_max = brainmask_indices[1].min(), brainmask_indices[1].max()
    z_min, z_max = brainmask_indices[2].min(), brainmask_indices[2].max()

    # Create empty array of original (full) shape
    restored_mask = np.zeros(brainmask.shape, dtype=np.float32)

    # Place the probability map back into the original image space
    restored_mask[x_min:x_max, y_min:y_max, z_min:z_max] = instance_mask

    return restored_mask


# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Undo brain masking on FLAMES probability maps."
)
parser.add_argument("flames_dir",    help="Directory containing FLAMES probability maps")
parser.add_argument("brainmask_dir", help="Directory containing brain masks")
parser.add_argument("output_dir",    help="Directory where restored images will be saved")
args = parser.parse_args()

flames_dir    = args.flames_dir
brainmask_dir = args.brainmask_dir
output_dir    = args.output_dir

os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Process each subject
# -----------------------------------------------------------------------------
subjects = sorted([
    x for x in os.listdir(flames_dir)
    if x.startswith("sub-") and x.endswith("pred-prob.nii.gz")
])

if len(subjects) == 0:
    print("No files matching sub-*_pred-prob.nii.gz found in", flames_dir)
    exit(1)

print(f"Found {len(subjects)} subject(s) to process.\n")

for subj in subjects:
    subject_id = subj.split("_")[0]
    print("Processing subject:", subject_id)

    # Load FLAMES probability map and brain mask
    instance_mask = nib.load(pjoin(flames_dir, subj)).get_fdata()
    brainmask_path = pjoin(brainmask_dir, f"{subject_id}_brainmask.nii.gz")

    if not os.path.exists(brainmask_path):
        print(f"  Skipping {subject_id}: brain mask not found at {brainmask_path}")
        continue

    brainmask_nib = nib.load(brainmask_path)
    brainmask = brainmask_nib.get_fdata()

    # Restore probability map to original image space
    restored_instance_mask = undo_brainmask(brainmask, instance_mask)

    # Save restored image using brain mask affine and float32 dtype
    restored_nib = nib.Nifti1Image(
        restored_instance_mask.astype(np.float32),
        brainmask_nib.affine
    )
    restored_nib.set_data_dtype(np.float32)
    nib.save(restored_nib, pjoin(output_dir, subj))

print("\nAll subjects processed.")