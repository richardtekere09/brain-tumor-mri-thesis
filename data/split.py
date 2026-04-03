"""
Task 0.6 — Generate data/splits.json from TextBraTS Train.json and Test.json.

Policy:
  - Train.json → hold out val_fraction as validation (random seed)
  - Test.json  → test set (unchanged)
  - Output: {"train": [...], "val": [...], "test": [...]}

Usage:
    python data/split.py \
        --train_json data/TextBraTS/Train.json \
        --test_json  data/TextBraTS/Test.json \
        --val_fraction 0.15 \
        --seed 42 \
        --output data/splits.json
"""
import argparse
import json
import logging
import math
import random
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _extract_ids(json_path: str) -> List[str]:
    """Return ordered list of case IDs from a TextBraTS JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    key = list(data.keys())[0]
    entries = data[key]
    ids: List[str] = []
    for entry in entries:
        label_path = entry.get("label", "")
        case_id = label_path.split("/")[0]
        if case_id:
            ids.append(case_id)
    return ids


def make_splits(
    train_json: str,
    test_json: str,
    val_fraction: float,
    seed: int,
    output: str,
) -> None:
    train_ids = _extract_ids(train_json)
    test_ids = _extract_ids(test_json)

    rng = random.Random(seed)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)

    n_val = math.ceil(len(shuffled) * val_fraction)
    val_ids = sorted(shuffled[:n_val])
    final_train_ids = sorted(shuffled[n_val:])
    final_test_ids = sorted(test_ids)

    splits = {
        "train": final_train_ids,
        "val": val_ids,
        "test": final_test_ids,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    total = len(final_train_ids) + len(val_ids) + len(final_test_ids)
    logger.info("Train cases : %d", len(final_train_ids))
    logger.info("Val cases   : %d  (%.1f%% of original train)", len(val_ids), 100 * len(val_ids) / len(train_ids))
    logger.info("Test cases  : %d", len(final_test_ids))
    logger.info("Total       : %d", total)
    logger.info("Splits written to: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data splits from TextBraTS JSONs.")
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    make_splits(
        train_json=args.train_json,
        test_json=args.test_json,
        val_fraction=args.val_fraction,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
