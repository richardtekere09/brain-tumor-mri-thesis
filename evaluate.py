"""
evaluate.py — Full test-set evaluation for all three models.

Usage:
    python evaluate.py --config configs/model_a.yaml --checkpoint checkpoints/model_a/best.pth
    python evaluate.py --config configs/model_b.yaml --checkpoint checkpoints/model_b/best.pth
    python evaluate.py --config configs/model_c.yaml --checkpoint checkpoints/model_c/best.pth

Metrics computed per case (WT, TC, ET):
    Dice, IoU, HD95

Model C also computes:
    Micro-F1 and exact-match on slot predictions

Outputs:
    results/evaluation_{model_name}.json   — mean ± std across test set
    Prints per-case and summary table to stdout
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# HD95 helper
# ─────────────────────────────────────────────────────────────────────────────

def _hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (voxel units).

    Returns 0 if both are empty, inf if exactly one is empty.
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    try:
        from scipy.ndimage import distance_transform_edt
        pred_b = pred.astype(bool)
        gt_b   = gt.astype(bool)
        # Distance from each gt voxel to nearest pred voxel
        dt_pred = distance_transform_edt(~pred_b)
        dt_gt   = distance_transform_edt(~gt_b)
        d1 = dt_pred[gt_b]
        d2 = dt_gt[pred_b]
        all_d = np.concatenate([d1, d2])
        return float(np.percentile(all_d, 95))
    except Exception:
        return float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Per-case metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_seg_metrics(pred_bin: torch.Tensor, label: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Args:
        pred_bin: [1, 3, H, W, D] binary float
        label:    [1, 3, H, W, D] binary float
    Returns:
        dict keyed by 'wt', 'tc', 'et' each with dice, iou, hd95
    """
    results = {}
    for ch, key in enumerate(["wt", "tc", "et"]):
        p = pred_bin[0, ch].cpu().numpy().astype(bool)
        t = label[0, ch].cpu().numpy().astype(bool)

        inter = (p & t).sum()
        union = (p | t).sum()
        dice  = float(2 * inter / (p.sum() + t.sum() + 1e-8))
        iou   = float(inter / (union + 1e-8))
        hd95  = _hd95(p, t)

        results[key] = {"dice": dice, "iou": iou, "hd95": hd95}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Slot metrics (Model C)
# ─────────────────────────────────────────────────────────────────────────────

def compute_slot_metrics(
    all_preds: List[np.ndarray],
    all_targets: List[np.ndarray],
) -> Dict[str, float]:
    """Micro-F1 and exact-match for 5-slot binary predictions."""
    preds   = np.array(all_preds)    # [N, 5]
    targets = np.array(all_targets)  # [N, 5]

    # Binarise slot 3 (burden) — any non-zero = 1 for micro-F1
    pred_bin   = (preds > 0.5).astype(int)
    target_bin = (targets > 0).astype(int)

    # Micro-F1 across all slots
    tp = (pred_bin & target_bin).sum()
    fp = (pred_bin & ~target_bin).sum()
    fn = (~pred_bin & target_bin).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    micro_f1  = float(2 * precision * recall / (precision + recall + 1e-8))

    # Exact match — all 5 slots correct
    exact = float((pred_bin == target_bin).all(axis=1).mean())

    return {"micro_f1": micro_f1, "exact_match": exact}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    from train import load_config, build_model

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # ── Build and load model ─────────────────────────────────────────────────
    model = build_model(cfg, device)
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info("Loaded checkpoint: %s  (epoch=%d  best_dice=%.4f)",
                args.checkpoint, ckpt.get("epoch", 0), ckpt.get("best_dice", 0))

    # ── Build test dataset ───────────────────────────────────────────────────
    from data.dataset import BraTSDataset
    from data.transforms import build_val_transforms

    load_text = (model_name == "vision_text")
    tokenizer = None
    if load_text:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["biobert_name"])

    test_ds = BraTSDataset(
        split="test",
        splits_file=cfg["data"]["splits_file"],
        brats_root=cfg["data"]["brats_root"],
        textbrats_root=cfg["data"]["textbrats_root"],
        transforms=build_val_transforms(),
        load_text=load_text,
        tokenizer=tokenizer,
    )
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    logger.info("Test cases: %d", len(test_ds))

    # ── Inference ────────────────────────────────────────────────────────────
    from monai.inferers import sliding_window_inference

    all_metrics: Dict[str, List[float]] = {
        f"{r}_{m}": [] for r in ["wt", "tc", "et"] for m in ["dice", "iou", "hd95"]
    }
    slot_preds_all, slot_targets_all = [], []

    for i, batch in enumerate(loader):
        case_id = batch["case_id"][0]
        image   = batch["image"].to(device)
        label   = batch["label"].to(device)

        t0 = time.time()

        def _predict(x):
            if model_name == "vision_text":
                seg, _, _, _ = model(
                    x,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                return seg
            return model(x)

        with torch.no_grad():
            pred_logits = sliding_window_inference(
                image, roi_size=(96, 96, 96), sw_batch_size=1,
                predictor=_predict, overlap=0.5,
            )

        infer_secs = time.time() - t0
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        seg_m = compute_seg_metrics(pred_bin, label)

        for key in ["wt", "tc", "et"]:
            for m in ["dice", "iou", "hd95"]:
                all_metrics[f"{key}_{m}"].append(seg_m[key][m])

        # Slot predictions (Model C)
        if model_name == "vision_text" and "slot_labels" in batch:
            with torch.no_grad():
                # Use a centre-crop 96³ patch for slot prediction
                h, w, d = image.shape[2], image.shape[3], image.shape[4]
                sh = max(0, (h - 96) // 2)
                sw_ = max(0, (w - 96) // 2)
                sd = max(0, (d - 96) // 2)
                img_crop = image[:, :, sh:sh+96, sw_:sw_+96, sd:sd+96]
                _, slot_logits, _, _ = model(
                    img_crop,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
            slot_pred   = (torch.sigmoid(slot_logits) > 0.5).float().cpu().numpy()[0]
            slot_target = batch["slot_labels"].numpy()[0]
            slot_preds_all.append(slot_pred)
            slot_targets_all.append(slot_target)

        logger.info(
            "[%3d/%d] %s | Dice WT=%.3f TC=%.3f ET=%.3f | "
            "IoU WT=%.3f TC=%.3f ET=%.3f | "
            "HD95 WT=%.1f TC=%.1f ET=%.1f | %.1fs  %.2fGB",
            i + 1, len(test_ds), case_id,
            seg_m["wt"]["dice"], seg_m["tc"]["dice"], seg_m["et"]["dice"],
            seg_m["wt"]["iou"],  seg_m["tc"]["iou"],  seg_m["et"]["iou"],
            seg_m["wt"]["hd95"], seg_m["tc"]["hd95"], seg_m["et"]["hd95"],
            infer_secs, gpu_mem,
        )

    # ── Aggregate results ────────────────────────────────────────────────────
    results = {}
    for region in ["WT", "TC", "ET"]:
        key = region.lower()
        results[region] = {
            "dice_mean": float(np.mean(all_metrics[f"{key}_dice"])),
            "dice_std":  float(np.std(all_metrics[f"{key}_dice"])),
            "iou_mean":  float(np.mean(all_metrics[f"{key}_iou"])),
            "iou_std":   float(np.std(all_metrics[f"{key}_iou"])),
            "hd95_mean": float(np.nanmean([x for x in all_metrics[f"{key}_hd95"] if np.isfinite(x)] or [0])),
            "hd95_std":  float(np.nanstd( [x for x in all_metrics[f"{key}_hd95"] if np.isfinite(x)] or [0])),
            "hd95_inf_count": int(sum(1 for x in all_metrics[f"{key}_hd95"] if not np.isfinite(x))),
        }

    results["mean_dice"] = float(np.mean([
        results["WT"]["dice_mean"],
        results["TC"]["dice_mean"],
        results["ET"]["dice_mean"],
    ]))

    if slot_preds_all:
        results["slot_metrics"] = compute_slot_metrics(slot_preds_all, slot_targets_all)

    results["meta"] = {
        "model":      model_name,
        "checkpoint": args.checkpoint,
        "test_cases": len(test_ds),
        "device":     str(device),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS — {model_name.upper()}")
    print("=" * 65)
    print(f"  {'Region':<8}  {'Dice':>12}  {'IoU':>12}  {'HD95':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for region in ["WT", "TC", "ET"]:
        r = results[region]
        hd_str = f"{r['hd95_mean']:.2f}±{r['hd95_std']:.2f}"
        if r["hd95_inf_count"] > 0:
            hd_str += f" ({r['hd95_inf_count']} inf)"
        print(f"  {region:<8}  "
              f"{r['dice_mean']:.4f}±{r['dice_std']:.4f}  "
              f"{r['iou_mean']:.4f}±{r['iou_std']:.4f}  "
              f"{hd_str}")
    print(f"\n  Mean Dice (WT+TC+ET): {results['mean_dice']:.4f}")

    if "slot_metrics" in results:
        sm = results["slot_metrics"]
        print(f"\n  Slot predictions (Model C):")
        print(f"    Micro-F1     : {sm['micro_f1']:.4f}")
        print(f"    Exact-match  : {sm['exact_match']:.4f}")

    print("=" * 65)

    # ── Save JSON ────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"evaluation_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--config",     required=True,  help="Path to model config yaml")
    parser.add_argument("--checkpoint", required=True,  help="Path to best.pth checkpoint")
    parser.add_argument("--device",     default=None,   help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
