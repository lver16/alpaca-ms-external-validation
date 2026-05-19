import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

import sys
sys.path.append("/conflunet/evaluation")

from conflunet.evaluation.utils import (
    match_instances,
    find_confluent_lesions,
    find_tierx_confluent_instances,
)

from conflunet.evaluation.metrics import compute_metrics


# ── Constants ─────────────────────────────────────────────────────────────────
APRL_DIR = Path("/linux/luverheyen/data/processed_50_custom/")
REF_DIR  = Path("/linux/luverheyen/data/labels_raw/")
OUT_DIR  = Path("/linux/luverheyen/data/aprl_metrics_with_wo_CLU_50/")

# Label ranges in the 3D reference mask
# PRL and regular lesions:
#   - Lesion evaluation: ALL labels >= 1000 (PRL 1000-1999 + regular lesions >= 2000)
#   - PRL evaluation:    only labels 1000–1999
LESION_LO = 1000
PRL_LO    = 1000
PRL_HI    = 1999

# Probability threshold for APRL PRL classification.
# A candidate is considered a rimpos (PRL) prediction if rimpos_proba > PRL_THRESHOLD.
PRL_THRESHOLD = 0.5

# ── PRL subject filter ────────────────────────────────────────────────────────
# "all"         → keep all subjects regardless of PRL presence
# "ref"         → keep only subjects with at least 1 PRL in the reference mask
# "pred"        → keep only subjects with at least 1 PRL in the prediction
# "ref_or_pred" → keep only subjects with at least 1 PRL in reference OR prediction
PRL_FILTER = "all"   # ← change this to "ref", "pred" or "ref_or_pred" as needed


# ── File loading ──────────────────────────────────────────────────────────────
def load_int_img(path: Path):
    img  = nib.load(str(path))
    data = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int32)
    return img, data


# ── Affine utilities ──────────────────────────────────────────────────────────
def is_las_ras_flip(affine1: np.ndarray, affine2: np.ndarray) -> bool:
    rotation_close = np.allclose(np.abs(affine1[:3, :3]), np.abs(affine2[:3, :3]), atol=1e-2)
    directly_close = np.allclose(affine1[:3, :3], affine2[:3, :3], atol=1e-2)
    return rotation_close and not directly_close


def check_center_alignment(img1, img2) -> float:
    center_vox = np.array(img1.shape) / 2
    center1    = img1.affine @ np.append(center_vox, 1)
    center2    = img2.affine @ np.append(center_vox, 1)
    return float(np.max(np.abs(center1[:3] - center2[:3])))


def reorient_to_reference(img_to_reorient, img_reference):
    ref_ornt        = io_orientation(img_reference.affine)
    src_ornt        = io_orientation(img_to_reorient.affine)
    transform       = ornt_transform(src_ornt, ref_ornt)
    reoriented_data = apply_orientation(img_to_reorient.get_fdata(dtype=np.float32), transform)
    reoriented_data = np.rint(reoriented_data).astype(np.int32)
    reoriented_img  = nib.Nifti1Image(reoriented_data, img_reference.affine, img_reference.header)
    return reoriented_img, reoriented_data


def align_to_reference(ref_img, moving_img, moving_data):
    """Align moving image to reference orientation."""
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
              f"— reorienting APRL mask to reference convention.")
        return reorient_to_reference(moving_img, ref_img)
    else:
        raise ValueError(
            f"Affine mismatch is NOT a simple LAS/RAS flip — registration may be needed.\n"
            f"REF affine:\n{ref_img.affine}\nAPRL affine:\n{moving_img.affine}"
        )


# ── CSV parsing ───────────────────────────────────────────────────────────────
def parse_predictions(csv_path: Path) -> pd.DataFrame:
    """
    Parse APRL aprl_preds.csv.
    Expected columns: rimneg, rimpos   (probabilities for each candidate)
    Row index (implicit or explicit) corresponds to the integer label in
    aprl_leslabels.nii.gz — candidate IDs start at 1.

    Returns DataFrame with columns: candidate_id, rimneg, rimpos
    """
    df = pd.read_csv(csv_path)

    # Normalise column names to lowercase, strip whitespace
    df.columns = df.columns.str.strip().str.lower()

    required = {"rimneg", "rimpos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"aprl_preds.csv is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # candidate_id = row position + 1 (labels in the NIfTI are 1-based)
    df.insert(0, "candidate_id", np.arange(1, len(df) + 1, dtype=int))

    return df[["candidate_id", "rimneg", "rimpos"]]


# ── Candidate ID selection ────────────────────────────────────────────────────
def get_lesion_candidate_ids(pred_df: pd.DataFrame) -> set:
    """
    Lesion evaluation: every candidate present in the CSV is a detected lesion
    (APRL only outputs candidates that passed the MIMoSA threshold, so all rows
    represent lesion detections regardless of PRL classification).
    """
    return set(pred_df["candidate_id"].tolist())


def get_prl_candidate_ids(pred_df: pd.DataFrame, threshold: float = PRL_THRESHOLD) -> set:
    """
    PRL evaluation: candidates where rimpos probability > threshold.
    """
    mask = pred_df["rimpos"] > threshold
    return set(pred_df.loc[mask, "candidate_id"].tolist())


# ── Build instance masks ──────────────────────────────────────────────────────
def build_ref_lesion_mask(ref_data: np.ndarray) -> np.ndarray:
    """
    Reference for lesion evaluation: ALL labels >= 1000.
    PRL (1000-1999) and regular lesions (>=2000) are both subtypes of lesion.
    """
    mask = np.zeros_like(ref_data, dtype=np.int32)
    mask[ref_data >= LESION_LO] = ref_data[ref_data >= LESION_LO]
    return mask


def build_ref_prl_mask(ref_data: np.ndarray) -> np.ndarray:
    """
    Reference for PRL evaluation: labels 1000–1999 only.
    A rimpos candidate overlapping a >=2000 ref label → FP automatically.
    """
    mask = np.zeros_like(ref_data, dtype=np.int32)
    cond = (ref_data >= PRL_LO) & (ref_data <= PRL_HI)
    mask[cond] = ref_data[cond]
    return mask


def build_pred_mask(aprl_data: np.ndarray, candidate_ids: set) -> np.ndarray:
    """
    Keep only voxels whose label is in candidate_ids.
    This is the bridge between the CSV classifications and the spatial mask.
    compute_metrics never sees the CSV — only this filtered array.
    """
    mask = np.zeros_like(aprl_data, dtype=np.int32)
    for cid in candidate_ids:
        mask[aprl_data == cid] = cid
    return mask


def get_confluent_ids(ref_mask: np.ndarray) -> set:
    """
    Return the union of CLU (tier 0) and CLU+ (tier 1), both with connectivity 6.

    - CLU  (tier 0, conn 6): lesions directly touching face-to-face
    - CLU+ (tier 1, conn 6): lesions near-touching after one dilation step

    tier 1 is always a superset of tier 0, so the union equals tier 1.
    Both are computed explicitly to make the intent clear.
    """
    cl_tier0 = set(find_confluent_lesions(ref_mask, connectivity=6))                     # CLU
    cl_tier1 = set(find_tierx_confluent_instances(ref_mask, tier=1, connectivity=6))     # CLU+
    return cl_tier0  # union — in practice equals cl_tier1


def remove_confluent_from_ref(ref_mask: np.ndarray, confluent_ids: set) -> np.ndarray:
    """
    Zero out all confluent lesion labels from the reference mask.
    Predicted instances overlapping those regions stay in instance_pred
    and will become FP automatically in compute_metrics.
    """
    ref_no_clu = np.copy(ref_mask)
    for cid in confluent_ids:
        ref_no_clu[ref_no_clu == cid] = 0
    return ref_no_clu


def subject_has_prl_ref(ref_data: np.ndarray) -> bool:
    """True if the reference mask contains at least one PRL label (1000-1999)."""
    return bool(np.any((ref_data >= PRL_LO) & (ref_data <= PRL_HI)))


def subject_has_prl_pred(pred_df: pd.DataFrame) -> bool:
    """True if APRL predicted at least one PRL candidate (rimpos > PRL_THRESHOLD)."""
    return bool((pred_df["rimpos"] > PRL_THRESHOLD).any())


# ── Per-subject processing ────────────────────────────────────────────────────
def process_subject(subject_id: str, ref_file: Path, subject_dir: Path):
    """
    subject_dir : /linux/luverheyen/data/processed_20_custom/<subject_id>/
    aprl outputs are expected inside subject_dir/aprl/
    """
    print(f"  [SUBJECT] {subject_id}")

    aprl_dir = subject_dir / "aprl"

    ref_img,  ref_data  = load_int_img(ref_file)
    aprl_img, aprl_data = load_int_img(aprl_dir / "aprl_leslabels.nii.gz")

    if ref_data.shape != aprl_data.shape:
        raise ValueError(f"Shape mismatch: ref={ref_data.shape} vs aprl={aprl_data.shape}")

    aprl_img, aprl_data = align_to_reference(ref_img, aprl_img, aprl_data)

    pred_df = parse_predictions(aprl_dir / "aprl_preds.csv")

    # ── PRL subject filter ────────────────────────────────────────────────────
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

    # ── Candidate ID selection ────────────────────────────────────────────────
    lesion_ids = get_lesion_candidate_ids(pred_df)           # all candidates = lesion detections
    prl_ids    = get_prl_candidate_ids(pred_df)              # rimpos > PRL_THRESHOLD

    print(f"    [APRL] {len(lesion_ids)} lesion pred (all candidates) | "
          f"{len(prl_ids)} PRL pred (rimpos > {PRL_THRESHOLD})")

    voxel_size = tuple(ref_img.header.get_zooms())

    # ── Build full reference masks ────────────────────────────────────────────
    ref_lesion = build_ref_lesion_mask(ref_data)   # >= 1000
    ref_prl    = build_ref_prl_mask(ref_data)      # 1000–1999

    # ── Build no-CLU reference masks ─────────────────────────────────────────
    # Confluent IDs are computed separately per mask type because a lesion
    # that is confluent in the full lesion mask might not be confluent in
    # the PRL-only mask (different neighbours after filtering)
    confluent_lesion_ids = get_confluent_ids(ref_lesion)
    confluent_prl_ids    = get_confluent_ids(ref_prl)

    ref_lesion_no_clu = remove_confluent_from_ref(ref_lesion, confluent_lesion_ids)
    ref_prl_no_clu    = remove_confluent_from_ref(ref_prl,    confluent_prl_ids)

    print(f"    [CLU] lesion: {len(confluent_lesion_ids)} confluent removed | "
          f"PRL: {len(confluent_prl_ids)} confluent removed")

    # ── Build prediction masks ────────────────────────────────────────────────
    pred_lesion = build_pred_mask(aprl_data, lesion_ids)
    pred_prl    = build_pred_mask(aprl_data, prl_ids)
    # Note: pred masks are the SAME for full and no-CLU evaluations.
    # Predicted instances over confluent ref regions stay in pred and
    # become FP automatically — this is intentional.

    # ── Precompute CLU ids on FULL lesion ref ─────────────────────────────────
    # CLU for PRL runs must be computed on ref_lesion (all lesions >= 1000),
    # not on ref_prl (1000-1999 only), because a PRL can be confluent with a
    # regular lesion — using ref_prl alone would miss those cases.
    cl_ids_full = {
        (0, 6):  find_confluent_lesions(ref_lesion, connectivity=6),
        (0, 26): find_confluent_lesions(ref_lesion, connectivity=26),
        (1, 6):  find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=6),
        (1, 26): find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=26),
    }

    results = {}

    # ── Run 1: Lesion (full) ──────────────────────────────────────────────────
    n_ref  = len(np.unique(ref_lesion[ref_lesion > 0]))
    n_pred = len(np.unique(pred_lesion[pred_lesion > 0]))
    print(f"    [LESION     ] ref={n_ref} | pred={n_pred}")
    if n_ref == 0 and n_pred == 0:
        print("    [LESION     ] Both empty — skipping.")
    else:
        m, _, _ = compute_metrics(pred_lesion, ref_lesion, voxel_size=voxel_size)
        results["lesion"] = m

    # ── Run 2: Lesion (no CLU/CLU+) ──────────────────────────────────────────
    n_ref_nc  = len(np.unique(ref_lesion_no_clu[ref_lesion_no_clu > 0]))
    n_pred_nc = len(np.unique(pred_lesion[pred_lesion > 0]))
    print(f"    [LESION NoCLU] ref={n_ref_nc} | pred={n_pred_nc}")
    if n_ref_nc == 0 and n_pred_nc == 0:
        print("    [LESION NoCLU] Both empty — skipping.")
    else:
        m, _, _ = compute_metrics(pred_lesion, ref_lesion_no_clu, voxel_size=voxel_size)
        results["lesion_no_clu"] = m

    # ── Run 3: PRL (full) ─────────────────────────────────────────────────────
    n_ref  = len(np.unique(ref_prl[ref_prl > 0]))
    n_pred = len(np.unique(pred_prl[pred_prl > 0]))
    print(f"    [PRL        ] ref={n_ref} | pred={n_pred}")
    if n_ref == 0 and n_pred == 0:
        print("    [PRL        ] Both empty — skipping.")
    else:
        m, _, _ = compute_metrics(pred_prl, ref_prl, voxel_size=voxel_size,
                                  precomputed_cl_ids=cl_ids_full)
        results["prl"] = m

    # ── Run 4: PRL (no CLU/CLU+) ─────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reference index: sub-XXX_ses-01_mask-instances.nii.gz  →  key = "sub-XXX_ses-01"
    ref_index = {}
    for p in sorted(REF_DIR.glob("sub-*_ses-*_mask-instances.nii.gz")):
        subject_id = "_".join(p.name.split("_")[:2])   # sub-XXX_ses-01
        ref_index[subject_id] = p

    # APRL subject index: one folder per subject inside APRL_DIR
    aprl_index = {
        p.name: p
        for p in sorted(APRL_DIR.iterdir())
        if p.is_dir() and p.name.startswith("sub-")
    }

    subjects = sorted(set(ref_index.keys()) & set(aprl_index.keys()))
    print(f"[INFO] {len(subjects)} paired subjects found.")

    # Four row collectors — one per evaluation type
    rows = {"lesion": [], "lesion_no_clu": [], "prl": [], "prl_no_clu": []}
    n_ok, n_fail, n_skip = 0, 0, 0

    for subject_id in subjects:
        print(f"\n[PROCESSING] {subject_id}")
        try:
            results = process_subject(
                subject_id,
                ref_index[subject_id],
                aprl_index[subject_id],
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

    # ── Save four Excel files ─────────────────────────────────────────────────
    output_files = {
        "lesion":        "metrics_lesion.xlsx",
        "lesion_no_clu": "metrics_lesion_no_clu.xlsx",
        "prl":           "metrics_prl.xlsx",
        "prl_no_clu":    "metrics_prl_no_clu.xlsx",
    }

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