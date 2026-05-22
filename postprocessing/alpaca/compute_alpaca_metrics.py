"""
compute_alpaca_metrics.py
-------------------------------------------------------------------------------
Description:
    Computes lesion and PRL segmentation metrics for ALPaCA predictions against
    reference instance label masks. For each subject, the script:
      1. Loads the ALPaCA labeled candidates mask and predictions CSV
      2. Aligns the ALPaCA mask to the reference image space if needed
      3. Builds lesion and PRL prediction/reference masks
      4. Optionally removes confluent lesions (CLU) from the reference
      5. Runs compute_metrics() for lesion and PRL evaluations
      6. Saves results to Excel files

    Reference label encoding:
      - Labels >= 1000       : all lesions (used for lesion evaluation)
      - Labels 1000 to 1999  : PRL lesions (used for PRL evaluation)
      - Labels >= 2000       : non-PRL lesions

    Prediction encoding (from predictions.csv):
      - V1=1 : lesion candidate
      - V2=1 : PRL candidate
      - V3=1 : CVS candidate

    For lesion evaluation: any candidate with V1=1 OR V2=1 OR V3=1 counts
    as a detected lesion, since PRL and CVS are subtypes of lesion.
    For PRL evaluation: only candidates with V2=1 are used.

Usage:
    python compute_alpaca_metrics.py --alpaca_dir <alpaca_dir>
                                     --ref_dir <ref_dir>
                                     --out_dir <out_dir>
                                     --conflunet_dir <conflunet_dir>
                                     [--prl_filter <prl_filter>]
                                     [--compute_no_clu]

Arguments:
    --alpaca_dir      Path to the ALPaCA output directory containing one
                      subfolder per subject
    --ref_dir         Path to the directory containing reference instance
                      label masks (*_ses-*_mask-instances.nii.gz)
    --out_dir         Path to the directory where output Excel files will
                      be saved
    --conflunet_dir   Path to the conflunet package directory (used for
                      sys.path to import conflunet.evaluation utilities)
    --prl_filter      Subject filter based on PRL presence (default: "ref"):
                        "all"         : keep all subjects
                        "ref"         : keep only subjects with PRL in reference
                        "pred"        : keep only subjects with PRL in prediction
                        "ref_or_pred" : keep subjects with PRL in ref OR pred
    --compute_no_clu  Also compute metrics after removing confluent lesions
                      from the reference (default: False)

Example:
    python compute_alpaca_metrics.py \\
        --alpaca_dir /data/alpaca_out \\
        --ref_dir /data/labels_raw \\
        --out_dir /data/alpaca_metrics \\
        --conflunet_dir /conflunet/evaluation

    python compute_alpaca_metrics.py \\
        --alpaca_dir /data/alpaca_out \\
        --ref_dir /data/labels_raw \\
        --out_dir /data/alpaca_metrics \\
        --conflunet_dir /conflunet/evaluation \\
        --prl_filter ref_or_pred --compute_no_clu

Outputs (saved in --out_dir):
    - metrics_lesion.xlsx         : lesion metrics per subject (full)
    - metrics_prl.xlsx            : PRL metrics per subject (full)
    - metrics_lesion_no_clu.xlsx  : lesion metrics without confluent lesions
                                    (only if --compute_no_clu is set)
    - metrics_prl_no_clu.xlsx     : PRL metrics without confluent lesions
                                    (only if --compute_no_clu is set)
"""

import argparse
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute lesion and PRL segmentation metrics for ALPaCA predictions."
)
parser.add_argument(
    "--alpaca_dir", required=True,
    help="Path to ALPaCA output directory (one subfolder per subject)"
)
parser.add_argument(
    "--ref_dir", required=True,
    help="Path to directory containing reference instance label masks"
)
parser.add_argument(
    "--out_dir", required=True,
    help="Path to directory where output Excel files will be saved"
)
parser.add_argument(
    "--conflunet_dir", required=True,
    help="Path to the conflunet package directory for importing evaluation utilities"
)
parser.add_argument(
    "--prl_filter",
    default="ref",
    choices=["all", "ref", "pred", "ref_or_pred"],
    help="Subject filter based on PRL presence (default: 'ref')"
)
parser.add_argument(
    "--compute_no_clu",
    action="store_true",
    help="Also compute metrics after removing confluent lesions from the reference"
)
args = parser.parse_args()

ALPACA_DIR     = Path(args.alpaca_dir)
REF_DIR        = Path(args.ref_dir)
OUT_DIR        = Path(args.out_dir)
PRL_FILTER     = args.prl_filter
COMPUTE_NO_CLU = args.compute_no_clu

# -----------------------------------------------------------------------------
# Validate input directories
# -----------------------------------------------------------------------------
for path, name in [(ALPACA_DIR, "alpaca_dir"), (REF_DIR, "ref_dir")]:
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path} ({name})")

# -----------------------------------------------------------------------------
# Import conflunet evaluation utilities
# -----------------------------------------------------------------------------
sys.path.append(args.conflunet_dir)

from conflunet.evaluation.utils import (
    match_instances,
    find_confluent_lesions,
    find_tierx_confluent_instances,
)
from conflunet.evaluation.metrics import compute_metrics

# -----------------------------------------------------------------------------
# Label range constants
# PRL and CVS are subtypes of lesion:
#   - Lesion evaluation: ALL labels >= 1000
#   - PRL evaluation:    labels 1000-1999 only
# -----------------------------------------------------------------------------
LESION_LO = 1000
PRL_LO    = 1000
PRL_HI    = 1999


# -----------------------------------------------------------------------------
# File loading utilities
# -----------------------------------------------------------------------------

def load_int_img(path: Path):
    """Load a NIfTI image and round to integer array."""
    img  = nib.load(str(path))
    data = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int32)
    return img, data


# -----------------------------------------------------------------------------
# Affine alignment utilities
# -----------------------------------------------------------------------------

def is_las_ras_flip(affine1: np.ndarray, affine2: np.ndarray) -> bool:
    """Check if two affines differ only by a LAS/RAS flip."""
    rotation_close  = np.allclose(np.abs(affine1[:3, :3]), np.abs(affine2[:3, :3]), atol=1e-2)
    directly_close  = np.allclose(affine1[:3, :3], affine2[:3, :3], atol=1e-2)
    return rotation_close and not directly_close


def check_center_alignment(img1, img2) -> float:
    """Return the maximum center coordinate difference between two images."""
    center_vox = np.array(img1.shape) / 2
    center1    = img1.affine @ np.append(center_vox, 1)
    center2    = img2.affine @ np.append(center_vox, 1)
    return float(np.max(np.abs(center1[:3] - center2[:3])))


def reorient_to_reference(img_to_reorient, img_reference):
    """Reorient an image to match the orientation of a reference image."""
    ref_ornt        = io_orientation(img_reference.affine)
    src_ornt        = io_orientation(img_to_reorient.affine)
    transform       = ornt_transform(src_ornt, ref_ornt)
    reoriented_data = apply_orientation(img_to_reorient.get_fdata(dtype=np.float32), transform)
    reoriented_data = np.rint(reoriented_data).astype(np.int32)
    reoriented_img  = nib.Nifti1Image(reoriented_data, img_reference.affine, img_reference.header)
    return reoriented_img, reoriented_data


def align_to_reference(ref_img, moving_img, moving_data):
    """
    Align moving image to reference orientation.
    Handles identical affines and LAS/RAS flips. Raises an error for other
    affine mismatches that would require registration.
    """
    if np.allclose(ref_img.affine, moving_img.affine, atol=1e-2):
        print("    [AFFINE] Identical — no reorientation needed.")
        return moving_img, moving_data
    elif is_las_ras_flip(ref_img.affine, moving_img.affine):
        max_diff = check_center_alignment(ref_img, moving_img)
        if max_diff > 5.0:
            raise ValueError(
                f"LAS/RAS flip detected but centers differ by {max_diff:.2f}mm "
                f"— files may not cover the same region."
            )
        print(f"    [AFFINE] LAS/RAS flip detected (center diff={max_diff:.2f}mm) "
              f"— reorienting ALPaCA mask to reference convention.")
        return reorient_to_reference(moving_img, ref_img)
    else:
        raise ValueError(
            f"Affine mismatch is NOT a simple LAS/RAS flip — registration may be needed.\n"
            f"REF affine:\n{ref_img.affine}\nALPaCA affine:\n{moving_img.affine}"
        )


# -----------------------------------------------------------------------------
# CSV parsing
# -----------------------------------------------------------------------------

def parse_predictions(csv_path: Path) -> pd.DataFrame:
    """
    Parse ALPaCA predictions.csv.
    Expected format:
        ,V1,V2,V3
        1,1,0,0
    Returns DataFrame with columns: candidate_id, V1, V2, V3.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "candidate_id"
    df = df.reset_index()
    df["candidate_id"] = df["candidate_id"].astype(int)
    return df


# -----------------------------------------------------------------------------
# Candidate ID selection
# -----------------------------------------------------------------------------

def get_lesion_candidate_ids(pred_df: pd.DataFrame) -> set:
    """
    Lesion evaluation: V1=1 OR V2=1 OR V3=1.
    Any candidate classified as anything counts as a detected lesion since
    PRL and CVS are subtypes of lesion. Rejected candidates (all zeros)
    become FN automatically if they overlap a reference lesion.
    """
    mask = (pred_df["V1"] == 1) | (pred_df["V2"] == 1) | (pred_df["V3"] == 1)
    return set(pred_df.loc[mask, "candidate_id"].tolist())


def get_prl_candidate_ids(pred_df: pd.DataFrame) -> set:
    """PRL evaluation: V2=1 regardless of V1 or V3."""
    mask = pred_df["V2"] == 1
    return set(pred_df.loc[mask, "candidate_id"].tolist())


# -----------------------------------------------------------------------------
# Reference and prediction mask builders
# -----------------------------------------------------------------------------

def build_ref_lesion_mask(ref_data: np.ndarray) -> np.ndarray:
    """
    Reference for lesion evaluation: all labels >= 1000.
    PRL (1000-1999) and regular lesions (>=2000) are both subtypes of lesion.
    """
    mask = np.zeros_like(ref_data, dtype=np.int32)
    mask[ref_data >= LESION_LO] = ref_data[ref_data >= LESION_LO]
    return mask


def build_ref_prl_mask(ref_data: np.ndarray) -> np.ndarray:
    """
    Reference for PRL evaluation: labels 1000-1999 only.
    A V2=1 candidate overlapping a >=2000 ref label becomes FP automatically.
    """
    mask = np.zeros_like(ref_data, dtype=np.int32)
    cond = (ref_data >= PRL_LO) & (ref_data <= PRL_HI)
    mask[cond] = ref_data[cond]
    return mask


def build_pred_mask(alpaca_data: np.ndarray, candidate_ids: set) -> np.ndarray:
    """
    Build a prediction mask keeping only voxels whose label is in candidate_ids.
    This bridges the CSV classifications and the spatial mask —
    compute_metrics never sees the CSV, only this filtered array.
    """
    mask = np.zeros_like(alpaca_data, dtype=np.int32)
    for cid in candidate_ids:
        mask[alpaca_data == cid] = cid
    return mask


# -----------------------------------------------------------------------------
# Confluent lesion utilities
# -----------------------------------------------------------------------------

def get_confluent_ids(ref_mask: np.ndarray) -> set:
    """
    Return confluent lesion IDs (CLU tier 0, connectivity 6).
    CLU: lesions directly touching face-to-face.
    """
    return set(find_confluent_lesions(ref_mask, connectivity=6))


def remove_confluent_from_ref(ref_mask: np.ndarray, confluent_ids: set) -> np.ndarray:
    """
    Zero out all confluent lesion labels from the reference mask.
    Predicted instances overlapping those regions stay in pred_mask
    and become FP automatically in compute_metrics.
    """
    ref_no_clu = np.copy(ref_mask)
    for cid in confluent_ids:
        ref_no_clu[ref_no_clu == cid] = 0
    return ref_no_clu


# -----------------------------------------------------------------------------
# PRL subject filter helpers
# -----------------------------------------------------------------------------

def subject_has_prl_ref(ref_data: np.ndarray) -> bool:
    """True if the reference mask contains at least one PRL label (1000-1999)."""
    return bool(np.any((ref_data >= PRL_LO) & (ref_data <= PRL_HI)))


def subject_has_prl_pred(pred_df: pd.DataFrame) -> bool:
    """True if ALPaCA predicted at least one PRL candidate (V2=1)."""
    return bool((pred_df["V2"] == 1).any())


# -----------------------------------------------------------------------------
# Per-subject processing
# -----------------------------------------------------------------------------

def process_subject(subject_id: str, ref_file: Path, alpaca_dir: Path) -> dict:
    """
    Run all metric evaluations for a single subject.
    Returns a dict with keys: lesion, prl, and optionally
    lesion_no_clu and prl_no_clu if COMPUTE_NO_CLU is True.
    Returns an empty dict if the subject is filtered out.
    """
    print(f"  [SUBJECT] {subject_id}")

    # Load images
    ref_img,    ref_data    = load_int_img(ref_file)
    alpaca_img, alpaca_data = load_int_img(alpaca_dir / "labeled_candidates.nii.gz")

    if ref_data.shape != alpaca_data.shape:
        raise ValueError(f"Shape mismatch: ref={ref_data.shape} vs alpaca={alpaca_data.shape}")

    # Align ALPaCA mask to reference space if needed
    alpaca_img, alpaca_data = align_to_reference(ref_img, alpaca_img, alpaca_data)

    pred_df = parse_predictions(alpaca_dir / "predictions.csv")

    # Apply PRL subject filter
    if PRL_FILTER != "all":
        has_ref  = subject_has_prl_ref(ref_data)
        has_pred = subject_has_prl_pred(pred_df)
        if PRL_FILTER == "ref" and not has_ref:
            print(f"    [FILTER] Skipped — no PRL in reference (filter=ref)")
            return {}
        if PRL_FILTER == "pred" and not has_pred:
            print(f"    [FILTER] Skipped — no PRL in prediction (filter=pred)")
            return {}
        if PRL_FILTER == "ref_or_pred" and not (has_ref or has_pred):
            print(f"    [FILTER] Skipped — no PRL in reference or prediction (filter=ref_or_pred)")
            return {}

    # Get candidate IDs from predictions CSV
    lesion_ids   = get_lesion_candidate_ids(pred_df)
    prl_ids      = get_prl_candidate_ids(pred_df)
    rejected_ids = set(pred_df.loc[
        (pred_df["V1"] == 0) & (pred_df["V2"] == 0) & (pred_df["V3"] == 0),
        "candidate_id"
    ].tolist())

    print(f"    [ALPACA] {len(lesion_ids)} lesion pred (V1|V2|V3=1) | "
          f"{len(prl_ids)} PRL pred (V2=1) | "
          f"{len(rejected_ids)} rejected (all zeros)")

    voxel_size = tuple(ref_img.header.get_zooms())

    # Build full reference masks
    ref_lesion = build_ref_lesion_mask(ref_data)
    ref_prl    = build_ref_prl_mask(ref_data)

    # Build no-CLU reference masks if needed
    if COMPUTE_NO_CLU:
        confluent_lesion_ids = get_confluent_ids(ref_lesion)
        confluent_prl_ids    = get_confluent_ids(ref_prl)
        ref_lesion_no_clu    = remove_confluent_from_ref(ref_lesion, confluent_lesion_ids)
        ref_prl_no_clu       = remove_confluent_from_ref(ref_prl,    confluent_prl_ids)
        print(f"    [CLU] lesion: {len(confluent_lesion_ids)} confluent removed | "
              f"PRL: {len(confluent_prl_ids)} confluent removed")

    # Precompute confluent IDs for all connectivity variants (used in PRL metrics)
    cl_ids_full = {
        (0, 6):  find_confluent_lesions(ref_lesion, connectivity=6),
        (0, 26): find_confluent_lesions(ref_lesion, connectivity=26),
        (1, 6):  find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=6),
        (1, 26): find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=26),
    }

    # Build prediction masks
    # Note: pred masks are the same for full and no-CLU evaluations —
    # predicted instances over confluent ref regions become FP automatically
    pred_lesion = build_pred_mask(alpaca_data, lesion_ids)
    pred_prl    = build_pred_mask(alpaca_data, prl_ids)

    results = {}

    # Run 1: Lesion (full)
    n_ref  = len(np.unique(ref_lesion[ref_lesion > 0]))
    n_pred = len(np.unique(pred_lesion[pred_lesion > 0]))
    print(f"    [LESION     ] ref={n_ref} | pred={n_pred}")
    if n_ref == 0 and n_pred == 0:
        print("    [LESION     ] Both empty — skipping.")
    else:
        m, _, _ = compute_metrics(pred_lesion, ref_lesion, voxel_size=voxel_size)
        results["lesion"] = m

    # Run 2: Lesion (no CLU)
    if COMPUTE_NO_CLU:
        n_ref_nc  = len(np.unique(ref_lesion_no_clu[ref_lesion_no_clu > 0]))
        n_pred_nc = len(np.unique(pred_lesion[pred_lesion > 0]))
        print(f"    [LESION NoCLU] ref={n_ref_nc} | pred={n_pred_nc}")
        if n_ref_nc == 0 and n_pred_nc == 0:
            print("    [LESION NoCLU] Both empty — skipping.")
        else:
            m, _, _ = compute_metrics(pred_lesion, ref_lesion_no_clu, voxel_size=voxel_size)
            results["lesion_no_clu"] = m

    # Run 3: PRL (full)
    n_ref  = len(np.unique(ref_prl[ref_prl > 0]))
    n_pred = len(np.unique(pred_prl[pred_prl > 0]))
    print(f"    [PRL        ] ref={n_ref} | pred={n_pred}")
    if n_ref == 0 and n_pred == 0:
        print("    [PRL        ] Both empty — skipping.")
    else:
        m, _, _ = compute_metrics(pred_prl, ref_prl, voxel_size=voxel_size,
                                  precomputed_cl_ids=cl_ids_full)
        results["prl"] = m

    # Run 4: PRL (no CLU)
    if COMPUTE_NO_CLU:
        n_ref_nc  = len(np.unique(ref_prl_no_clu[ref_prl_no_clu > 0]))
        n_pred_nc = len(np.unique(pred_prl[pred_prl > 0]))
        print(f"    [PRL   NoCLU] ref={n_ref_nc} | pred={n_pred_nc}")
        if n_ref_nc == 0 and n_pred_nc == 0:
            print("    [PRL   NoCLU] Both empty — skipping.")
        else:
            m, _, _ = compute_metrics(pred_prl, ref_prl_no_clu, voxel_size=voxel_size,
                                      precomputed_cl_ids=cl_ids_full)
            results["prl_no_clu"] = m

    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build subject index from reference and ALPaCA directories
    ref_index = {
        p.name.split("_ses-")[0]: p
        for p in sorted(REF_DIR.glob("sub-*_ses-*_mask-instances.nii.gz"))
    }

    alpaca_index = {
        p.name: p
        for p in sorted(ALPACA_DIR.iterdir())
        if p.is_dir() and p.name.startswith("sub-")
    }

    # Process only subjects present in both directories
    subjects = sorted(set(ref_index.keys()) & set(alpaca_index.keys()))
    print(f"[INFO] {len(subjects)} paired subjects found.")

    # Pre-allocate row collectors for each evaluation type
    rows = {"lesion": [], "prl": []}
    if COMPUTE_NO_CLU:
        rows.update({"lesion_no_clu": [], "prl_no_clu": []})

    n_ok, n_fail, n_skip = 0, 0, 0

    for subject_id in subjects:
        print(f"\n[PROCESSING] {subject_id}")
        try:
            results = process_subject(
                subject_id,
                ref_index[subject_id],
                alpaca_index[subject_id],
            )

            if not results:
                n_skip += 1
                continue

            for key in rows:
                if key in results:
                    rows[key].append({"subject": subject_id, **results[key]})

            n_ok += 1

        except Exception as e:
            print(f"  [FAIL] {subject_id}: {e}")
            n_fail += 1

    # Save one Excel file per evaluation type
    output_files = {"lesion": "metrics_lesion.xlsx", "prl": "metrics_prl.xlsx"}
    if COMPUTE_NO_CLU:
        output_files.update({
            "lesion_no_clu": "metrics_lesion_no_clu.xlsx",
            "prl_no_clu":    "metrics_prl_no_clu.xlsx",
        })

    for key, filename in output_files.items():
        if rows[key]:
            df = pd.DataFrame(rows[key])
            out_path = OUT_DIR / filename
            df.to_excel(out_path, index=False)
            print(f"[EXCEL] {filename} saved ({len(df)} subjects)")
        else:
            print(f"[EXCEL] {filename} — no data to save.")

    print(f"\n[DONE] OK={n_ok} | Skipped={n_skip} | Failed={n_fail}")


if __name__ == "__main__":
    main()