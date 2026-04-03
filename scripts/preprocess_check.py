"""
scripts/preprocess_check.py — End-to-end preprocessing verification for one BraTS case.

Runs: load → crop → norm → clip → remap → patch
Prints a summary of shapes, value ranges, and label stats.
Saves a PNG to results/preprocess_check.png showing 3 axial slices:
  T1ce channel | WT mask | ET mask

Usage:
    python scripts/preprocess_check.py \
        --case BraTS20_Training_001 \
        --brats_root data/BraTS2020_TrainingData \
        --splits_file data/splits.json
"""
import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import _find_nii
from data.transforms import build_train_transforms, build_val_transforms

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

_MODALITIES = ["t1", "t1ce", "t2", "flair"]


def run_check(case: str, brats_root: str, output_path: str) -> None:
    case_dir = os.path.join(brats_root, case)

    # ------------------------------------------------------------------ load
    logger.info("Loading case: %s", case)
    image_raw = np.stack([
        nib.load(_find_nii(case_dir, f"{case}_{m}")).get_fdata()
        for m in _MODALITIES
    ]).astype(np.float32)
    label_raw = nib.load(_find_nii(case_dir, f"{case}_seg")).get_fdata().astype(np.int32)

    logger.info("Raw image shape : %s", image_raw.shape)
    logger.info("Raw label shape : %s", label_raw.shape)
    logger.info("Raw label values: %s", np.unique(label_raw).tolist())

    # ------------------------------------------------- run train transforms
    sample = {"image": image_raw, "label": label_raw}
    transforms = build_train_transforms(patch_size=(96, 96, 96))
    out = transforms(sample)

    image: torch.Tensor = out["image"]   # [4, 96, 96, 96]
    label: torch.Tensor = out["label"]   # [3, 96, 96, 96]

    # ----------------------------------------------------------- print stats
    logger.info("─" * 50)
    logger.info("After transforms:")
    logger.info("  image shape  : %s", tuple(image.shape))
    logger.info("  label shape  : %s", tuple(label.shape))
    logger.info("  image min    : %.4f", image.min().item())
    logger.info("  image max    : %.4f", image.max().item())
    logger.info("  image mean   : %.4f", image.mean().item())
    logger.info("  NaN in image : %s", torch.isnan(image).any().item())
    logger.info("  Inf in image : %s", torch.isinf(image).any().item())
    logger.info("  WT voxels    : %d", int(label[0].sum()))
    logger.info("  TC voxels    : %d", int(label[1].sum()))
    logger.info("  ET voxels    : %d", int(label[2].sum()))

    assert not torch.isnan(image).any(), "NaN detected in image after transforms"
    assert not torch.isinf(image).any(), "Inf detected in image after transforms"
    assert image.min() >= -5.1 and image.max() <= 5.1, "Values outside [-5,5] clip range"

    # --------------------------------------------------------- save PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Pick the axial slice with the most ET voxels so all regions are visible.
    # Fall back to the WT-richest slice if ET is absent.
    et_per_slice = label[2].sum(dim=(0, 1))   # sum over H, W → [D]
    wt_per_slice = label[0].sum(dim=(0, 1))

    best_z = int(et_per_slice.argmax()) if et_per_slice.max() > 0 else int(wt_per_slice.argmax())
    logger.info("Visualising axial slice z=%d (max ET voxels=%d)", best_z, int(et_per_slice[best_z]))

    t1ce = image[1, :, :, best_z].numpy()   # T1ce channel [H, W]
    wt   = label[0, :, :, best_z].numpy()   # WT mask
    tc   = label[1, :, :, best_z].numpy()   # TC mask
    et   = label[2, :, :, best_z].numpy()   # ET mask

    # Normalise T1ce to [0, 1] for display
    t1ce_norm = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{case} — axial slice z={best_z}  |  WT={int(wt.sum())} TC={int(tc.sum())} ET={int(et.sum())} voxels",
                 fontsize=11)

    # Panel 1 — raw T1ce
    axes[0].imshow(t1ce_norm.T, cmap="gray", origin="lower")
    axes[0].set_title("T1ce (raw)")
    axes[0].axis("off")

    # Panel 2 — T1ce + WT overlay (red, semi-transparent)
    axes[1].imshow(t1ce_norm.T, cmap="gray", origin="lower")
    axes[1].imshow(wt.T, cmap="Reds", alpha=0.45, origin="lower", vmin=0, vmax=1)
    axes[1].set_title("T1ce + WT (red)")
    axes[1].axis("off")

    # Panel 3 — T1ce + TC overlay (yellow)
    axes[2].imshow(t1ce_norm.T, cmap="gray", origin="lower")
    axes[2].imshow(tc.T, cmap="YlOrBr", alpha=0.5, origin="lower", vmin=0, vmax=1)
    axes[2].set_title("T1ce + TC (yellow)")
    axes[2].axis("off")

    # Panel 4 — T1ce + ET overlay (blue) — should align with bright enhancing areas
    axes[3].imshow(t1ce_norm.T, cmap="gray", origin="lower")
    axes[3].imshow(et.T, cmap="Blues", alpha=0.55, origin="lower", vmin=0, vmax=1)
    axes[3].set_title("T1ce + ET (blue)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    logger.info("PNG saved → %s", output_path)
    logger.info("─" * 50)
    logger.info("preprocess_check PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end preprocessing check.")
    parser.add_argument("--case", required=True, help="BraTS case ID, e.g. BraTS20_Training_001")
    parser.add_argument("--brats_root", default="data/BraTS2020_TrainingData")
    parser.add_argument("--splits_file", default="data/splits.json")
    parser.add_argument("--output", default="results/preprocess_check.png")
    args = parser.parse_args()

    run_check(
        case=args.case,
        brats_root=args.brats_root,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
