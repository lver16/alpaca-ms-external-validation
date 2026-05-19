"""
fix_predictions.py

For each subject in alpaca_out/:
  1. Creates an archive/ subfolder (only if it doesn't already exist)
  2. Copies the original predictions.csv and probabilities.csv to archive/
  3. Fixes a missing index column if present:
       Correct format  → first column is unnamed (Unnamed: 0) with values 1,2,3,...
       Bad format      → no index column at all; pandas assigns 0-based integers
  4. Fixes discordant predictions in predictions.csv:
       - V2=1 (PRL) but V1=0 (no Lesion) → set V2 to 0
       - V3=1 (CVS) but V1=0 (no Lesion) → set V3 to 0
  5. Saves the corrected files in place

Expected correct format (predictions_sub_001.csv template):
    ,V1,V2,V3
    1,1,1,0
    2,1,0,0
    ...

Bad format (missing index column — pandas auto-assigns 0,1,2,...):
    "V1","V2","V3"
    0,0,0
    ...

Usage:
    python fix_predictions.py
    python fix_predictions.py --alpaca_dir /linux/luverheyen/data/alpaca_out
"""

import pandas as pd
import shutil
import argparse
from pathlib import Path

PRED_COLS = ["V1", "V2", "V3"]
PROB_COLS = ["Lesion", "PRL", "CVS"]


def needs_index_fix(df_raw: pd.DataFrame, expected_cols: list) -> bool:
    """
    Returns True when the CSV was saved without an index column, i.e.
    pandas read it with auto 0-based integer index and the columns match
    exactly the expected data columns (no leading unnamed column).
    """
    return list(df_raw.columns) == expected_cols


def load_and_fix_index(path: Path, expected_cols: list) -> tuple:
    """
    Load a CSV and ensure it has a proper 1-based integer index.

    - If the file has the correct format (Unnamed: 0 col + data cols):
        read with index_col=0 -> already correct.
    - If the file is missing the index column (columns == expected_cols):
        read normally, then assign index = range(1, n+1).

    Returns (df, index_was_fixed).
    """
    df_raw = pd.read_csv(path)

    if needs_index_fix(df_raw, expected_cols):
        # Missing index: assign 1-based integers
        df_raw.index = range(1, len(df_raw) + 1)
        df_raw.index.name = None
        return df_raw, True
    else:
        # Correct format: promote the Unnamed col to the index
        df = pd.read_csv(path, index_col=0)
        df.index.name = None
        return df, False


def archive_file(src: Path, archive_dir: Path) -> bool:
    """Copy src into archive_dir only if the archive copy does not exist yet."""
    archive_path = archive_dir / src.name
    if not archive_path.exists():
        archive_dir.mkdir(exist_ok=True)
        shutil.copy2(src, archive_path)
        return True
    return False


# ---------------------------------------------------------------------------
# predictions.csv
# ---------------------------------------------------------------------------

def fix_predictions(subject_dir: Path, archive_dir: Path, dry_run: bool) -> dict:
    pred_path = subject_dir / "predictions.csv"

    if not pred_path.exists():
        print(f"    predictions.csv  - SKIP (not found)")
        return {"status": "skipped", "index_fixed": False,
                "prl_fixed": 0, "cvs_fixed": 0, "n_fixed": 0}

    df, index_was_fixed = load_and_fix_index(pred_path, PRED_COLS)
    df.columns = PRED_COLS  # ensure column names are canonical

    # Count discordant rows BEFORE fix
    prl_discordant = int(((df["V2"] == 1) & (df["V1"] == 0)).sum())
    cvs_discordant = int(((df["V3"] == 1) & (df["V1"] == 0)).sum())
    n_fixed = prl_discordant + cvs_discordant

    if not dry_run:
        archive_file(pred_path, archive_dir)

        # Fix discordant predictions
        df.loc[(df["V2"] == 1) & (df["V1"] == 0), "V2"] = 0
        df.loc[(df["V3"] == 1) & (df["V1"] == 0), "V3"] = 0

        df.to_csv(pred_path, index=True)

    status = "fixed" if (index_was_fixed or n_fixed > 0) else "clean"
    print(f"    predictions.csv  - {status}"
          f"  |  index added: {index_was_fixed}"
          f"  |  PRL discordant: {prl_discordant}"
          f"  |  CVS discordant: {cvs_discordant}"
          f"  |  total zeroed: {n_fixed}"
          + (" [DRY RUN]" if dry_run else ""))

    return {"status": status, "index_fixed": index_was_fixed,
            "prl_fixed": prl_discordant, "cvs_fixed": cvs_discordant,
            "n_fixed": n_fixed}


# ---------------------------------------------------------------------------
# probabilities.csv
# ---------------------------------------------------------------------------

def fix_probabilities(subject_dir: Path, archive_dir: Path, dry_run: bool) -> dict:
    prob_path = subject_dir / "probabilities.csv"

    if not prob_path.exists():
        print(f"    probabilities.csv - SKIP (not found)")
        return {"status": "skipped", "index_fixed": False}

    df, index_was_fixed = load_and_fix_index(prob_path, PROB_COLS)

    if not dry_run:
        archive_file(prob_path, archive_dir)

        if index_was_fixed:
            df.to_csv(prob_path, index=True)

    status = "fixed" if index_was_fixed else "clean"
    print(f"    probabilities.csv - {status}"
          f"  |  index added: {index_was_fixed}"
          + (" [DRY RUN]" if dry_run else ""))

    return {"status": status, "index_fixed": index_was_fixed}


# ---------------------------------------------------------------------------
# Per-subject entry point
# ---------------------------------------------------------------------------

def fix_subject(subject_dir: Path, dry_run: bool = False) -> dict:
    subject_id  = subject_dir.name
    archive_dir = subject_dir / "archive"

    print(f"  [{subject_id}]")

    pred_result = fix_predictions(subject_dir, archive_dir, dry_run)
    prob_result = fix_probabilities(subject_dir, archive_dir, dry_run)

    return {
        "subject":          subject_id,
        "pred_status":      pred_result["status"],
        "pred_index_fixed": pred_result.get("index_fixed", False),
        "prl_fixed":        pred_result.get("prl_fixed", 0),
        "cvs_fixed":        pred_result.get("cvs_fixed", 0),
        "n_fixed":          pred_result.get("n_fixed", 0),
        "prob_status":      prob_result["status"],
        "prob_index_fixed": prob_result.get("index_fixed", False),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(alpaca_dir: str, dry_run: bool):
    root = Path(alpaca_dir)
    if not root.exists():
        print(f"ERROR: directory not found: {root}")
        return

    subject_dirs = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])

    if not subject_dirs:
        print(f"No subject folders found in {root}")
        return

    print(f"Found {len(subject_dirs)} subjects"
          + (" - DRY RUN (no files will be modified)" if dry_run else ""))
    print()

    results = []
    for subject_dir in subject_dirs:
        results.append(fix_subject(subject_dir, dry_run=dry_run))
        print()

    def count(results, key, val):
        return sum(1 for r in results if r.get(key) == val)

    print("=" * 60)
    print(f"DONE - {len(subject_dirs)} subjects processed")
    print()
    print("  predictions.csv")
    print(f"    Fixed (index or discordant): {count(results, 'pred_status', 'fixed')}")
    print(f"    Clean:                       {count(results, 'pred_status', 'clean')}")
    print(f"    Skipped (not found):         {count(results, 'pred_status', 'skipped')}")
    print(f"    Index column added:          {sum(1 for r in results if r.get('pred_index_fixed'))}")
    print(f"    Total PRL candidates zeroed: {sum(r.get('prl_fixed', 0) for r in results)}")
    print(f"    Total CVS candidates zeroed: {sum(r.get('cvs_fixed', 0) for r in results)}")
    print()
    print("  probabilities.csv")
    print(f"    Fixed (index):       {count(results, 'prob_status', 'fixed')}")
    print(f"    Clean:               {count(results, 'prob_status', 'clean')}")
    print(f"    Skipped (not found): {count(results, 'prob_status', 'skipped')}")
    print(f"    Index column added:  {sum(1 for r in results if r.get('prob_index_fixed'))}")
    if not dry_run:
        print()
        print("  Original files archived in each subject's archive/ folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix index column and discordant predictions in ALPaCA output files.")
    parser.add_argument(
        "--alpaca_dir",
        default="/linux/luverheyen/data/alpaca_out",
        help="Root directory with one subfolder per subject")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be changed without modifying any file")
    args = parser.parse_args()
    main(args.alpaca_dir, args.dry_run)