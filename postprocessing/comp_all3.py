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

# Import compute_metrics — adjust to the actual module filename if different
from conflunet.evaluation.metrics import compute_metrics


# ── Constants ─────────────────────────────────────────────────────────────────
ALPACA_DIR = Path("/linux/luverheyen/data/alpaca_out/")
REF_DIR    = Path("/linux/luverheyen/data/labels_raw/")
OUT_DIR    = Path("/linux/luverheyen/data/alpaca_metrics_with_wo_CLU_flames_PRL_ref/")

COMPUTE_NO_CLU = False   # ← set to False to skip no-CLU evaluations

# Label ranges in the 3D reference patch
# PRL and CVS are subtypes of lesion, so:
#   - Lesion evaluation: ALL labels >= 1000 (PRL 1000-1999 + regular lesions >= 2000)
#   - PRL evaluation:    only labels 1000–1999
LESION_LO = 1000
PRL_LO    = 1000
PRL_HI    = 1999

# ── PRL subject filter ────────────────────────────────────────────────────────
# "all"         → keep all subjects regardless of PRL presence
# "ref"         → keep only subjects with at least 1 PRL in the reference mask
# "pred"        → keep only subjects with at least 1 PRL in the prediction
# "ref_or_pred" → keep only subjects with at least 1 PRL in reference OR prediction
PRL_FILTER = "ref"   # ← change this to "ref", "pred" or "ref_or_pred" as needed


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
              f"— reorienting ALPaCA mask to reference convention.")
        return reorient_to_reference(moving_img, ref_img)
    else:
        raise ValueError(
            f"Affine mismatch is NOT a simple LAS/RAS flip — registration may be needed.\n"
            f"REF affine:\n{ref_img.affine}\nALPaCA affine:\n{moving_img.affine}"
        )


# ── CSV parsing ───────────────────────────────────────────────────────────────
def parse_predictions(csv_path: Path) -> pd.DataFrame:
    """
    Parse ALPaCA predictions.csv.
    Header:  ,V1,V2,V3
    Rows:    1001,1,0,0
    Returns DataFrame with columns: candidate_id, V1, V2, V3
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "candidate_id"
    df = df.reset_index()
    df["candidate_id"] = df["candidate_id"].astype(int)
    return df


# ── Candidate ID selection ────────────────────────────────────────────────────
def get_lesion_candidate_ids(pred_df: pd.DataFrame) -> set:
    """
    Lesion evaluation: V1=1 OR V2=1 OR V3=1.
    Any candidate classified as anything counts as a detected lesion since
    PRL and CVS are both subtypes of lesion.
    Rejected candidates (all zeros) are excluded — they become FN automatically
    if they overlap a reference lesion.
    """
    mask = (pred_df["V1"] == 1) | (pred_df["V2"] == 1) | (pred_df["V3"] == 1)
    return set(pred_df.loc[mask, "candidate_id"].tolist())


def get_prl_candidate_ids(pred_df: pd.DataFrame) -> set:
    """
    PRL evaluation: V2=1 regardless of V1 or V3.
    """
    mask = pred_df["V2"] == 1
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
    A V2=1 candidate overlapping a >=2000 ref label → FP automatically.
    """
    mask = np.zeros_like(ref_data, dtype=np.int32)
    cond = (ref_data >= PRL_LO) & (ref_data <= PRL_HI)
    mask[cond] = ref_data[cond]
    return mask


def build_pred_mask(alpaca_data: np.ndarray, candidate_ids: set) -> np.ndarray:
    """
    Keep only voxels whose label is in candidate_ids.
    This is the bridge between the Excel classifications and the spatial mask.
    compute_metrics never sees the Excel — only this filtered array.
    """
    mask = np.zeros_like(alpaca_data, dtype=np.int32)
    for cid in candidate_ids:
        mask[alpaca_data == cid] = cid
    return mask


def get_confluent_ids(ref_mask: np.ndarray) -> set:
    """
    Return the union of CLU (tier 0) and CLU+ (tier 1), both with connectivity 6.

    - CLU  (tier 0, conn 6): lesions directly touching face-to-face
    - CLU+ (tier 1, conn 6): lesions near-touching after one dilation step

    tier 1 is always a superset of tier 0, so the union equals tier 1.
    Both are computed explicitly to make the intent clear.
    """
    cl_tier0 = set(find_confluent_lesions(ref_mask, connectivity=6))           # CLU
    cl_tier1 = set(find_tierx_confluent_instances(ref_mask, tier=1, connectivity=6))  # CLU+
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
    """True if the reference mask contains at least one PRL label (1000–1999)."""
    return bool(np.any((ref_data >= PRL_LO) & (ref_data <= PRL_HI)))


def subject_has_prl_pred(pred_df: pd.DataFrame) -> bool:
    """True if ALPaCA predicted at least one PRL candidate (V2=1)."""
    return bool((pred_df["V2"] == 1).any())


# ── Per-subject processing ────────────────────────────────────────────────────
def process_subject(subject_id: str, ref_file: Path, alpaca_dir: Path):
    print(f"  [SUBJECT] {subject_id}")

    ref_img, ref_data = load_int_img(ref_file)
    alpaca_img, alpaca_data = load_int_img(alpaca_dir / "labeled_candidates.nii.gz")

    if ref_data.shape != alpaca_data.shape:
        raise ValueError(f"Shape mismatch: ref={ref_data.shape} vs alpaca={alpaca_data.shape}")

    alpaca_img, alpaca_data = align_to_reference(ref_img, alpaca_img, alpaca_data)

    pred_df = parse_predictions(alpaca_dir / "predictions.csv")

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
    lesion_ids   = get_lesion_candidate_ids(pred_df)   # V1|V2|V3=1
    prl_ids      = get_prl_candidate_ids(pred_df)      # V2=1

    rejected_ids = set(
        pred_df.loc[
            (pred_df["V1"] == 0) & (pred_df["V2"] == 0) & (pred_df["V3"] == 0),
            "candidate_id"
        ].tolist()
    )
    print(f"    [ALPACA] {len(lesion_ids)} lesion pred (V1|V2|V3=1) | "
          f"{len(prl_ids)} PRL pred (V2=1) | "
          f"{len(rejected_ids)} rejected (all zeros)")

    voxel_size = tuple(ref_img.header.get_zooms())

    # ── Build full reference masks ────────────────────────────────────────────
    ref_lesion = build_ref_lesion_mask(ref_data)   # >= 1000
    ref_prl    = build_ref_prl_mask(ref_data)      # 1000–1999

    # ── Build no-CLU reference masks ─────────────────────────────────────────
    if COMPUTE_NO_CLU:
        confluent_lesion_ids = get_confluent_ids(ref_lesion)
        confluent_prl_ids    = get_confluent_ids(ref_prl)

        ref_lesion_no_clu = remove_confluent_from_ref(ref_lesion, confluent_lesion_ids)
        ref_prl_no_clu    = remove_confluent_from_ref(ref_prl,    confluent_prl_ids)

        print(f"    [CLU] lesion: {len(confluent_lesion_ids)} confluent removed | "
              f"PRL: {len(confluent_prl_ids)} confluent removed")

    cl_ids_full = {
        (0, 6):  find_confluent_lesions(ref_lesion, connectivity=6),
        (0, 26): find_confluent_lesions(ref_lesion, connectivity=26),
        (1, 6):  find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=6),
        (1, 26): find_tierx_confluent_instances(ref_lesion, tier=1, connectivity=26),
    }

    # ── Build prediction masks ────────────────────────────────────────────────
    pred_lesion = build_pred_mask(alpaca_data, lesion_ids)
    pred_prl    = build_pred_mask(alpaca_data, prl_ids)
    # Note: pred masks are the SAME for full and no-CLU evaluations.
    # Predicted instances over confluent ref regions stay in pred and
    # become FP automatically — this is intentional.

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
    if COMPUTE_NO_CLU:
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
        m, _, _ = compute_metrics(pred_prl, ref_prl, voxel_size=voxel_size, precomputed_cl_ids=cl_ids_full)
        results["prl"] = m

    # ── Run 4: PRL (no CLU/CLU+) ─────────────────────────────────────────────
    if COMPUTE_NO_CLU:
        n_ref_nc  = len(np.unique(ref_prl_no_clu[ref_prl_no_clu > 0]))
        n_pred_nc = len(np.unique(pred_prl[pred_prl > 0]))
        print(f"    [PRL   NoCLU] ref={n_ref_nc} | pred={n_pred_nc}")
        if n_ref_nc == 0 and n_pred_nc == 0:
            print("    [PRL   NoCLU] Both empty — skipping.")
        else:
            m, _, _ = compute_metrics(pred_prl, ref_prl_no_clu, voxel_size=voxel_size, precomputed_cl_ids=cl_ids_full)
            results["prl_no_clu"] = m

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_index = {}
    for p in sorted(REF_DIR.glob("sub-*_ses-*_mask-instances.nii.gz")):
        subject_id = p.name.split("_ses-")[0]
        ref_index[subject_id] = p

    alpaca_index = {
        p.name: p
        for p in sorted(ALPACA_DIR.iterdir())
        if p.is_dir() and p.name.startswith("sub-")
    }

    subjects = sorted(set(ref_index.keys()) & set(alpaca_index.keys()))
    print(f"[INFO] {len(subjects)} paired subjects found.")

    # Four row collectors — one per evaluation type
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

    # ── Save Excel files ──────────────────────────────────────────────────────
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