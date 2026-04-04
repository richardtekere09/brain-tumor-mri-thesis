"""
scripts/visualize_model_a.py — Visual evaluation of ResUNet-18

For each test case: 3-row figure (slices at 25/50/75% of tumour z-range), 7 columns:
  T1ce | GT seg | Prediction | GT+Pred overlay | Grad-CAM | CAM overlay | GT on CAM
"""

import sys, os, argparse, pathlib, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage import measure
import torch
import torch.nn as nn
import nibabel as nib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "phase2_model_a"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BG = "#0d0d0d"


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM hooked on final_conv (96³ full resolution) for crisp spatial maps."""

    def __init__(self, model):
        self.model = model
        self._feats = self._grads = None
        # Hook the full-resolution decoder conv [B, 16, 96, 96, 96]
        self._h1 = model.final_conv.register_forward_hook(
            lambda m, i, o: setattr(self, "_feats", o.detach()))
        self._h2 = model.final_conv.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def remove(self):
        self._h1.remove(); self._h2.remove()

    def compute(self, image, target_channel=0):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(image)
        # Backward on the *positive* predicted voxels only (focus on where model
        # is confident, not background gradient noise)
        prob = torch.sigmoid(logits[0, target_channel])
        score = (prob * (prob > 0.3).float()).sum()
        score.backward()
        # Global-average-pool gradients over spatial dims → channel weights
        w = self._grads.mean(dim=(2, 3, 4), keepdim=True)  # [1,16,1,1,1]
        cam = torch.relu((w * self._feats).sum(dim=1))      # [1,96,96,96]
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        if cam.max() > 0: cam /= cam.max()
        return cam


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def _find_nii(folder, stem):
    for ext in (".nii", ".nii.gz"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"{folder}/{stem}")


def load_and_preprocess(case_id, brats_root, patch_size=(96, 96, 96)):
    from data.dataset import remap_labels
    d = os.path.join(brats_root, case_id)
    mods = ["t1", "t1ce", "t2", "flair"]
    image = np.stack([
        nib.load(_find_nii(d, f"{case_id}_{m}")).get_fdata()
        for m in mods]).astype(np.float32)
    seg = nib.load(_find_nii(d, f"{case_id}_seg")).get_fdata().astype(np.int32)

    t1ce_raw = image[1].copy()

    # z-score normalise
    img = image.copy()
    for c in range(img.shape[0]):
        mask = img[c] > 0
        if mask.sum() > 0:
            img[c][mask] = (img[c][mask] - img[c][mask].mean()) / (img[c][mask].std() + 1e-8)
    img = np.clip(img, -5, 5)

    label = remap_labels(seg)   # [3, H, W, D]

    # Find tumour centre (WT)
    wt = label[0]              # [H, W, D]
    if wt.sum() > 0:
        coords = np.argwhere(wt > 0)
        centre = coords.mean(axis=0).astype(int)
    else:
        centre = np.array([s // 2 for s in img.shape[1:]])

    # Crop 96³ patch centred on tumour
    H, W, D = img.shape[1], img.shape[2], img.shape[3]
    ph = patch_size
    starts = [
        int(max(0, min(centre[i] - ph[i] // 2, img.shape[i+1] - ph[i])))
        for i in range(3)
    ]
    sl = tuple(slice(starts[i], starts[i] + ph[i]) for i in range(3))

    img_patch   = img[:, sl[0], sl[1], sl[2]]        # [4, 96, 96, 96]
    label_patch = label[:, sl[0], sl[1], sl[2]]      # [3, 96, 96, 96]
    t1ce_patch  = t1ce_raw[sl[0], sl[1], sl[2]]      # [96, 96, 96]

    img_t   = torch.from_numpy(img_patch).float().unsqueeze(0)
    label_t = torch.from_numpy(label_patch).float()
    return img_t, label_t, t1ce_patch


def pick_three_slices(label_t):
    """
    Return 3 axial z-indices at 25/50/75 percentile of WT z-range.
    label_t: [3, H, W, D]
    """
    wt = label_t[0].numpy()           # [H, W, D]
    per_z = wt.sum(axis=(0, 1))       # [D]  — sum over H, W
    active = np.where(per_z > 0)[0]
    if len(active) == 0:
        D = wt.shape[2]
        return [D//4, D//2, 3*D//4]
    lo, hi = active[0], active[-1]
    z25 = int(lo + (hi - lo) * 0.25)
    z50 = int(lo + (hi - lo) * 0.50)
    z75 = int(lo + (hi - lo) * 0.75)
    return [z25, z50, z75]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def norm2d(arr):
    a = arr - arr.min()
    return a / (a.max() + 1e-8)


def seg_rgba(wt, tc, et, alpha=0.7):
    h, w = wt.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[wt & ~tc] = [0.0, 0.6, 0.7, alpha]   # teal  — edema
    rgba[tc & ~et] = [1.0, 0.5, 0.0, alpha]   # orange — core
    rgba[et]       = [1.0, 0.93, 0.0, alpha]  # yellow — ET
    return rgba


def draw_contours(ax, mask, color, lw=1.5):
    if mask.sum() == 0: return
    try:
        for c in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)
    except Exception:
        pass


def sax(ax, title, tc="white"):
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_title(title, color=tc, fontsize=8, pad=3)


def dice_ch(pred, gt):
    i = (pred * gt).sum()
    return float(2 * i / (pred.sum() + gt.sum() + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(case_id, img_t, label_t, pred_bin, cam, t1ce_raw):
    slices = pick_three_slices(label_t)   # [z25, z50, z75]

    # Overall Dice (3D)
    d_wt = dice_ch(pred_bin[0], label_t[0])
    d_tc = dice_ch(pred_bin[1], label_t[1])
    d_et = dice_ch(pred_bin[2], label_t[2])

    NCOLS = 7
    NROWS = 3
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"ResUNet-18  |  {case_id}  |  "
        f"3D Dice:  WT={d_wt:.3f}   TC={d_tc:.3f}   ET={d_et:.3f}",
        color="white", fontsize=12, fontweight="bold", y=1.01
    )

    col_titles = [
        "T1ce Input",
        "Ground Truth\n(teal=WT  orange=TC  yellow=ET)",
        "Prediction\n(same colours)",
        "GT + Pred Overlay\n(green=GT contour  red=Pred)",
        "Grad-CAM\n(WT channel, final_conv)",
        "Grad-CAM\nOverlay on T1ce",
        "GT Contour\non Grad-CAM",
    ]
    col_colors = ["white","white","#7ecfff","white","#ff7070","#ff7070","#88ff88"]

    kw = dict(interpolation="nearest", aspect="equal")

    for row, z in enumerate(slices):
        t1ce_s  = norm2d(t1ce_raw[:, :, z])
        t1ce_ns = norm2d(img_t[0, 1, :, :, z].numpy())
        cam_s   = cam[:, :, z]

        gt_wt = label_t[0, :, :, z].numpy() > 0.5
        gt_tc = label_t[1, :, :, z].numpy() > 0.5
        gt_et = label_t[2, :, :, z].numpy() > 0.5
        pr_wt = pred_bin[0, :, :, z].numpy() > 0.5
        pr_tc = pred_bin[1, :, :, z].numpy() > 0.5
        pr_et = pred_bin[2, :, :, z].numpy() > 0.5

        row_label = f"z={z}"

        # Col 0: T1ce raw
        ax = axes[row, 0]
        ax.imshow(t1ce_s, cmap="gray", **kw)
        sax(ax, col_titles[0] + f"\n({row_label})" if row == 0 else f"({row_label})")

        # Col 1: Ground truth
        ax = axes[row, 1]
        ax.imshow(t1ce_s, cmap="gray", **kw)
        ax.imshow(seg_rgba(gt_wt, gt_tc, gt_et), **kw)
        sax(ax, col_titles[1] if row == 0 else "")

        # Col 2: Prediction
        ax = axes[row, 2]
        ax.imshow(t1ce_s, cmap="gray", **kw)
        ax.imshow(seg_rgba(pr_wt, pr_tc, pr_et), **kw)
        d2 = dice_ch(pred_bin[0, :, :, z].numpy(), label_t[0, :, :, z].numpy())
        sax(ax, (col_titles[2] if row == 0 else "") + f"\nSlice Dice(WT)={d2:.3f}",
            tc="#7ecfff")

        # Col 3: GT+Pred overlay
        ax = axes[row, 3]
        ax.imshow(t1ce_s, cmap="gray", **kw)
        pred_any = pr_wt | pr_tc | pr_et
        red = np.zeros((*pred_any.shape, 4), dtype=np.float32)
        red[pred_any] = [0.9, 0.2, 0.2, 0.45]
        ax.imshow(red, **kw)
        draw_contours(ax, gt_wt, "#00ff88")
        draw_contours(ax, gt_tc, "#ffaa00")
        draw_contours(ax, gt_et, "#ffff00")
        sax(ax, col_titles[3] if row == 0 else "")

        # Col 4: Grad-CAM only
        ax = axes[row, 4]
        ax.imshow(np.zeros_like(t1ce_s), cmap="gray", **kw)
        ax.imshow(cam_s, cmap="jet", vmin=0, vmax=1, **kw)
        sax(ax, col_titles[4] if row == 0 else "", tc="#ff7070")

        # Col 5: Grad-CAM on T1ce
        ax = axes[row, 5]
        ax.imshow(t1ce_s, cmap="gray", **kw)
        ax.imshow(cam_s, cmap="jet", alpha=0.55, vmin=0, vmax=1, **kw)
        sax(ax, col_titles[5] if row == 0 else "", tc="#ff7070")

        # Col 6: GT contour on Grad-CAM
        ax = axes[row, 6]
        ax.imshow(np.zeros_like(t1ce_s), cmap="gray", **kw)
        ax.imshow(cam_s, cmap="jet", vmin=0, vmax=1, **kw)
        draw_contours(ax, gt_wt, "#00ff88")
        draw_contours(ax, gt_tc, "#ffaa00")
        draw_contours(ax, gt_et, "#ffff00")
        sax(ax, col_titles[6] if row == 0 else "", tc="#88ff88")

    legend = [
        Patch(color=(0.0, 0.6, 0.7),  label="WT edema"),
        Patch(color=(1.0, 0.5, 0.0),  label="TC core"),
        Patch(color=(1.0, 0.93, 0.0), label="ET enhancing"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               facecolor="#1a1a1a", labelcolor="white",
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/model_a/best.pth")
    parser.add_argument("--brats_root",  default="data/BraTS2020_TrainingData")
    parser.add_argument("--splits_file", default="data/splits.json")
    parser.add_argument("--num_cases",   type=int, default=3)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    from models.model_a import ResUNet18
    model = ResUNet18(in_channels=4, out_channels=3).to(device)
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded: epoch={ckpt['epoch']}  best_dice={ckpt.get('best_dice',0):.4f}")

    with open(args.splits_file) as f:
        test_cases = json.load(f)["test"][:args.num_cases]
    print(f"Cases: {test_cases}")

    gradcam = GradCAM(model)

    for case_id in test_cases:
        print(f"\n{case_id}")
        img_t, label_t, t1ce_raw = load_and_preprocess(case_id, args.brats_root)
        img_t = img_t.to(device)

        model.eval()
        with torch.no_grad():
            logits = model(img_t)
        pred_bin = (torch.sigmoid(logits) > 0.5).float().squeeze(0).cpu()

        cam = gradcam.compute(img_t, target_channel=0)

        fig = make_figure(case_id, img_t.cpu(), label_t, pred_bin, cam, t1ce_raw)
        out = OUT_DIR / f"visual_{case_id}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {out}")

    gradcam.remove()
    print(f"\nAll PNGs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
