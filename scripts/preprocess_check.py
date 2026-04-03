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

    # Choose 3 informative axial slices spread across the tumor depth.
    # Use z-indices where WT is present, then sample at 25 / 50 / 75 percentile.
    wt_per_slice = label[0].sum(dim=(0, 1))   # [D]
    et_per_slice = label[2].sum(dim=(0, 1))   # [D]
    tumor_zs = torch.where(wt_per_slice > 0)[0]

    if len(tumor_zs) >= 3:
        z_low  = int(tumor_zs[len(tumor_zs) // 4])
        z_mid  = int(tumor_zs[len(tumor_zs) // 2])
        z_high = int(tumor_zs[3 * len(tumor_zs) // 4])
    else:
        z_mid  = int(wt_per_slice.argmax())
        z_low  = max(0, z_mid - 10)
        z_high = min(image.shape[3] - 1, z_mid + 10)

    slices = [z_low, z_mid, z_high]
    slice_labels = ["lower", "middle", "upper"]
    logger.info("Visualising slices z=%s", slices)

    # Helper: build a colour-coded segmentation RGB image from binary channels.
    # WT-only (edema) = green, TC-only = orange, ET = magenta
    def _seg_rgb(wt_sl: np.ndarray, tc_sl: np.ndarray, et_sl: np.ndarray) -> np.ndarray:
        H, W = wt_sl.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        edema = (wt_sl > 0) & (tc_sl == 0)          # WT - TC
        core  = (tc_sl > 0) & (et_sl == 0)           # TC - ET
        rgb[edema] = [0.2, 0.8, 0.2]                 # green  — edema
        rgb[core]  = [1.0, 0.5, 0.0]                 # orange — necrotic core
        rgb[et_sl > 0] = [0.9, 0.1, 0.9]             # magenta — enhancing tumour
        return rgb

    # 3 rows × 5 columns: T1ce | T1ce+WT | T1ce+TC | T1ce+ET | Seg mask
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    col_titles = ["T1ce (raw)", "T1ce + WT (red)", "T1ce + TC (orange)", "T1ce + ET (blue)", "Seg mask"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row, (z, zlabel) in enumerate(zip(slices, slice_labels)):
        t1ce = image[1, :, :, z].numpy()
        wt   = label[0, :, :, z].numpy()
        tc   = label[1, :, :, z].numpy()
        et   = label[2, :, :, z].numpy()

        t1ce_norm = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
        seg_rgb   = _seg_rgb(wt, tc, et)

        row_info = f"z={z} ({zlabel})  WT={int(wt.sum())} TC={int(tc.sum())} ET={int(et.sum())}"
        axes[row, 0].set_ylabel(row_info, fontsize=8, rotation=0, labelpad=110, va="center")

        # Col 0 — raw T1ce
        axes[row, 0].imshow(t1ce_norm.T, cmap="gray", origin="lower")
        axes[row, 0].axis("off")

        # Col 1 — T1ce + WT
        axes[row, 1].imshow(t1ce_norm.T, cmap="gray", origin="lower")
        axes[row, 1].imshow(wt.T, cmap="Reds", alpha=0.45, origin="lower", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        # Col 2 — T1ce + TC
        axes[row, 2].imshow(t1ce_norm.T, cmap="gray", origin="lower")
        axes[row, 2].imshow(tc.T, cmap="YlOrBr", alpha=0.5, origin="lower", vmin=0, vmax=1)
        axes[row, 2].axis("off")

        # Col 3 — T1ce + ET
        axes[row, 3].imshow(t1ce_norm.T, cmap="gray", origin="lower")
        axes[row, 3].imshow(et.T, cmap="Blues", alpha=0.55, origin="lower", vmin=0, vmax=1)
        axes[row, 3].axis("off")

        # Col 4 — colour-coded seg mask (green=edema, orange=core, magenta=ET)
        axes[row, 4].imshow(t1ce_norm.T, cmap="gray", origin="lower")
        axes[row, 4].imshow(seg_rgb.transpose(1, 0, 2), alpha=0.65, origin="lower")
        axes[row, 4].axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=(0.2, 0.8, 0.2), label="Edema (WT−TC)"),
        Patch(facecolor=(1.0, 0.5, 0.0), label="Necrotic core (TC−ET)"),
        Patch(facecolor=(0.9, 0.1, 0.9), label="Enhancing tumour (ET)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"{case} — preprocessing check  |  patch {tuple(image.shape[1:])}",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
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
