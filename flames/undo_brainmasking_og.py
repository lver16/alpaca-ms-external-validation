import nibabel as nib
import numpy as np
import os
from os.path import join as pjoin

def undo_brainmask(brainmask, instance_mask):
    """
    Undo brain masking on an instance mask.

    Parameters:
    - brainmask (numpy.ndarray): A binary brain mask where non-brain regions are 0, sized [H_orig, W_orig, D_orig].
    - instance_mask (numpy.ndarray): An instance segmentation mask that has been brain-masked, sized [H_masked, W_masked, D_masked].

    Returns:
    - numpy.ndarray: The instance segmentation mask with non brain regions restored, sized [H_orig, W_orig, D_orig].
    """
    # find bbox of brainmask
    brainmask_indices = np.where(brainmask > 0)
    x_min, x_max = brainmask_indices[0].min(), brainmask_indices[0].max()
    y_min, y_max = brainmask_indices[1].min(), brainmask_indices[1].max()
    z_min, z_max = brainmask_indices[2].min(), brainmask_indices[2].max()
    # create empty array of original shape
    restored_mask = np.zeros(brainmask.shape, dtype=np.float32) #uint32 to float32
    # place instance_mask back into original shape
    restored_mask[x_min:x_max, y_min:y_max, z_min:z_max] = instance_mask
    # restored_mask[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = instance_mask.astype(np.float32) # to keep the last voxel in eahc axis
    return restored_mask


MAIN_FOLDER = "/linux/luverheyen/data/"
folders = ["flames_acls"]

for f in folders:
    print("\n\nProcessing folder:", f)
    subjects = sorted([x for x in os.listdir(pjoin(MAIN_FOLDER, f)) if x.startswith("sub-") and x.endswith("pred-prob.nii.gz")])
    for subj in subjects:
        subject_id = subj.split("_")[0]
        print("Processing subject:", subject_id)

        instance_mask = nib.load(pjoin(MAIN_FOLDER, f, subj)).get_fdata()
        brainmask_nib = nib.load(pjoin("/linux/luverheyen/data/synthstrip_raw/", f"{subject_id}_brainmask.nii.gz"))
        brainmask = brainmask_nib.get_fdata()
        restored_instance_mask = undo_brainmask(brainmask, instance_mask)
        # restored_nib = nib.Nifti1Image(restored_instance_mask, brainmask_nib.affine, brainmask_nib.header)
        restored_nib = nib.Nifti1Image(
            restored_instance_mask.astype(np.float32),
            brainmask_nib.affine
        )
        restored_nib.set_data_dtype(np.float32)
        nib.save(restored_nib, pjoin(MAIN_FOLDER, f, subj))

# instance_mask = nib.load("/home/mwynen/data/prl_instances_study/flames_acls/sub-005_ses-01_mask-instances.nii.gz").get_fdata()
# brainmask_nib = nib.load("/home/mwynen/data/cusl_wml/all/synthstrip_raw/sub-005_brainmask.nii.gz")
# brainmask = brainmask_nib.get_fdata()
# restored_instance_mask = undo_brainmask(brainmask, instance_mask)
# restored_nib = nib.Nifti1Image(restored_instance_mask, brainmask_nib.affine, brainmask_nib.header)
# nib.save(restored_nib, "/home/mwynen/data/prl_instances_study/flames_acls/sub-005_ses-01_mask-instances-restored.nii.gz")