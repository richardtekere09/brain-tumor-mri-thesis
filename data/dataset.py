"""
data/dataset.py — BraTS dataset class and label utilities.
"""
import json
import logging
import os
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Modality load order: T1, T1ce, T2, FLAIR
_MODALITIES = ["t1", "t1ce", "t2", "flair"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_nii(folder: str, stem: str) -> str:
    """Return path to stem.nii or stem.nii.gz, whichever exists."""
    for ext in (".nii", ".nii.gz"):
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"NIfTI file not found: {os.path.join(folder, stem)}(.nii|.nii.gz)")


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Convert raw BraTS segmentation to 3 binary channels (WT, TC, ET).

    Args:
        seg: Integer array of shape [H, W, D] with values in {0, 1, 2, 4}.

    Returns:
        Float32 array of shape [3, H, W, D]:
            Channel 0 (WT): 1.0 where label ∈ {1, 2, 4}
            Channel 1 (TC): 1.0 where label ∈ {1, 4}
            Channel 2 (ET): 1.0 where label == 4
    """
    wt = np.isin(seg, [1, 2, 4]).astype(np.float32)
    tc = np.isin(seg, [1, 4]).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


# ---------------------------------------------------------------------------
# Slot label computation (Task 4.3)
# ---------------------------------------------------------------------------

def compute_slot_labels(label: torch.Tensor) -> torch.Tensor:
    """Derive 5 report-slot labels from a remapped label tensor [3, H, W, D].

    Slots:
        0 — WT present  (binary)
        1 — TC present  (binary)
        2 — ET present  (binary)
        3 — tumor burden: 0=small (<1% brain vol), 1=medium, 2=large (>5%)
        4 — enhancement: 0=limited, 1=prominent (ET/WT ratio > 0.3)
    """
    wt_count = float(label[0].sum())
    tc_count = float(label[1].sum())
    et_count = float(label[2].sum())

    brain_voxels = float(label[0].numel())  # total spatial voxels

    wt_present = 1.0 if wt_count > 0 else 0.0
    tc_present = 1.0 if tc_count > 0 else 0.0
    et_present = 1.0 if et_count > 0 else 0.0

    wt_frac = wt_count / brain_voxels if brain_voxels > 0 else 0.0
    if wt_frac < 0.01:
        burden = 0.0
    elif wt_frac > 0.05:
        burden = 2.0
    else:
        burden = 1.0

    et_wt_ratio = et_count / wt_count if wt_count > 0 else 0.0
    enhancement = 1.0 if et_wt_ratio > 0.3 else 0.0

    return torch.tensor([wt_present, tc_present, et_present, burden, enhancement],
                        dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """PyTorch Dataset for BraTS 2020 + optional TextBraTS text features.

    Args:
        split:           One of 'train', 'val', 'test'.
        splits_file:     Path to data/splits.json.
        brats_root:      Root directory of BraTS cases (each case is a sub-folder).
        textbrats_root:  Root directory containing per-case text feature files
                         (expects <case_id>/<case_id>_flair_text.npy).
        transforms:      Callable applied to {'image': ndarray, 'label': ndarray}.
        load_text:       If True, also load and return tokenized text features.
        tokenizer:       HuggingFace tokenizer (required when load_text=True).
        max_text_length: Max token length for the tokenizer.
    """

    def __init__(
        self,
        split: str,
        splits_file: str,
        brats_root: str,
        textbrats_root: str,
        transforms: Optional[Callable] = None,
        load_text: bool = False,
        tokenizer=None,
        max_text_length: int = 512,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.brats_root = brats_root
        self.textbrats_root = textbrats_root
        self.transforms = transforms
        self.load_text = load_text
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

        with open(splits_file, encoding="utf-8") as f:
            splits = json.load(f)
        all_ids: List[str] = splits[split]

        # Filter out cases with missing files so DataLoader workers never crash
        self.case_ids = []
        skipped = []
        for cid in all_ids:
            case_dir = os.path.join(brats_root, cid)
            ok = True
            for stem in [f"{cid}_{m}" for m in _MODALITIES] + [f"{cid}_seg"]:
                found = any(
                    os.path.exists(os.path.join(case_dir, stem + ext))
                    for ext in (".nii", ".nii.gz")
                )
                if not found:
                    ok = False
                    break
            if ok:
                self.case_ids.append(cid)
            else:
                skipped.append(cid)

        if skipped:
            logger.warning("Skipped %d cases with missing files: %s", len(skipped), skipped)
        logger.info("BraTSDataset split=%s  cases=%d (skipped=%d)",
                    split, len(self.case_ids), len(skipped))

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict:
        case_id = self.case_ids[idx]
        case_dir = os.path.join(self.brats_root, case_id)

        # Load 4 modalities → [4, H, W, D]
        image = np.stack([
            nib.load(_find_nii(case_dir, f"{case_id}_{m}")).get_fdata()
            for m in _MODALITIES
        ]).astype(np.float32)

        # Load segmentation → [H, W, D]
        label = nib.load(_find_nii(case_dir, f"{case_id}_seg")).get_fdata().astype(np.int32)

        sample: Dict = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        sample["case_id"] = case_id

        if self.load_text:
            sample.update(self._load_text(case_id))

        return sample

    def _load_text(self, case_id: str) -> Dict:
        """Load text report and tokenize it."""
        txt_path = os.path.join(self.textbrats_root, case_id, f"{case_id}_flair_text.txt")
        if not os.path.exists(txt_path):
            # Fallback: use a generic placeholder so training can still proceed
            logger.warning("Text file missing for %s, using empty string.", case_id)
            text = ""
        else:
            with open(txt_path, encoding="utf-8") as f:
                text = f.read().strip()

        if self.tokenizer is None:
            raise ValueError("load_text=True requires a tokenizer.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_text_length,
            truncation=True,
            padding="max_length",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
