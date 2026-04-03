"""
Task 0.5 — Verify that every case ID in Train.json and Test.json
has both a BraTS MRI folder and a TextBraTS text annotation.

Handles both .nii and .nii.gz file extensions.

Usage:
    python data/verify_merge.py \
        --brats_root data/BraTS2020_TrainingData \
        --textbrats_root data/TextBraTS_hf/TextBraTSData \
        --train_json data/TextBraTS/Train.json \
        --test_json data/TextBraTS/Test.json
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _find_nii(folder: str, case_id: str, suffix: str) -> bool:
    """Return True if case_id_suffix.nii or case_id_suffix.nii.gz exists."""
    for ext in (".nii.gz", ".nii"):
        if os.path.exists(os.path.join(folder, f"{case_id}_{suffix}{ext}")):
            return True
    return False


def extract_case_ids(json_path: str) -> List[str]:
    """Return sorted list of unique case IDs from a TextBraTS JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # JSON is {"training": [...]} regardless of train/test
    key = list(data.keys())[0]
    entries = data[key]

    ids: List[str] = []
    for entry in entries:
        # label path is e.g. "BraTS20_Training_002/BraTS20_Training_002_seg.nii.gz"
        label_path = entry.get("label", "")
        case_id = label_path.split("/")[0]
        if case_id:
            ids.append(case_id)
    return ids


def verify(
    brats_root: str,
    textbrats_root: str,
    train_json: str,
    test_json: str,
) -> bool:
    train_ids = extract_case_ids(train_json)
    test_ids = extract_case_ids(test_json)
    all_ids = train_ids + test_ids

    logger.info("Train.json cases : %d", len(train_ids))
    logger.info("Test.json cases  : %d", len(test_ids))
    logger.info("Total IDs in JSON files: %d", len(all_ids))

    missing_mri: List[str] = []
    missing_text: List[str] = []
    matched: List[str] = []

    modalities = ["t1", "t1ce", "t2", "flair"]

    for case_id in all_ids:
        mri_dir = os.path.join(brats_root, case_id)
        mri_ok = os.path.isdir(mri_dir) and all(
            _find_nii(mri_dir, case_id, m) for m in modalities
        )
        if not mri_ok:
            missing_mri.append(case_id)

        text_file = os.path.join(textbrats_root, case_id, f"{case_id}_flair_text.npy")
        text_ok = os.path.exists(text_file)
        if not text_ok:
            missing_text.append(case_id)

        if mri_ok and text_ok:
            matched.append(case_id)

    logger.info("Fully matched cases:     %d", len(matched))
    logger.info("Missing MRI folders:     %d", len(missing_mri))
    logger.info("Missing text files:      %d", len(missing_text))

    if missing_mri:
        logger.error("Cases missing MRI: %s", missing_mri[:10])
    if missing_text:
        logger.error("Cases missing text: %s", missing_text[:10])

    if not missing_mri and not missing_text:
        logger.info("Verification PASSED.")
        return True
    else:
        logger.error("Verification FAILED.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BraTS + TextBraTS merge.")
    parser.add_argument("--brats_root", required=True)
    parser.add_argument("--textbrats_root", required=True)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    args = parser.parse_args()

    passed = verify(
        brats_root=args.brats_root,
        textbrats_root=args.textbrats_root,
        train_json=args.train_json,
        test_json=args.test_json,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
