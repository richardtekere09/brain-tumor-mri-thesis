"""
train.py — Unified training script for all three models.

Usage (CPU verification, 1 epoch, first 2 cases):
    python train.py --config configs/model_a.yaml --max_epochs 1 --num_samples 2 --device cpu

Full training on Kaggle GPU:
    python train.py --config configs/model_a.yaml

Auto-resume: if {save_dir}/last.pth exists, training continues from saved epoch.
"""
import argparse
import csv
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
import yaml

from data.dataset import BraTSDataset
from data.transforms import build_train_transforms, build_val_transforms
from losses.losses import SegmentationLoss, ModelCLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str) -> Dict:
    """Load model config and merge with base.yaml defaults."""
    cfg = _load_yaml(config_path)
    base_key = "_base_"
    if base_key in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop(base_key))
        base = _load_yaml(base_path)
        # Deep merge: model config overrides base
        merged = _deep_merge(base, cfg)
        return merged
    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict, device: torch.device) -> nn.Module:
    model_name = cfg["model"]["name"]
    logger.info("Building model: %s", model_name)

    if model_name == "resunet18":
        from models.model_a import ResUNet18, load_medicalnet_weights
        model = ResUNet18(
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
        )
        weights = cfg["model"].get("medicalnet_weights")
        if weights and os.path.exists(weights):
            load_medicalnet_weights(model, weights)
        else:
            logger.warning("MedicalNet weights not found at: %s — training from scratch", weights)

    elif model_name == "swin_unetr":
        from models.model_b import SwinUNETRWrapper
        model = SwinUNETRWrapper(
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
            feature_size=cfg["model"].get("feature_size", 48),
            use_checkpoint=cfg["model"].get("use_checkpoint", True),
            pretrained_weights=cfg["model"].get("pretrained_weights"),
        )

    elif model_name == "vision_text":
        from models.model_c import VisionTextModel
        model = VisionTextModel(
            in_channels=cfg["model"]["in_channels"],
            out_channels=cfg["model"]["out_channels"],
            feature_size=cfg["model"].get("feature_size", 48),
            swin_weights=cfg["model"].get("swin_weights"),
            biobert_name=cfg["model"].get("biobert_name", "dmis-lab/biobert-v1.1"),
            freeze_biobert=cfg["model"].get("freeze_biobert", True),
            fusion_dim=cfg["model"].get("fusion_dim", 768),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model.to(device)


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    t = cfg["training"]
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=t["lr"],
        weight_decay=t["weight_decay"],
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=cfg["training"]["scheduler_patience"],
        factor=0.5,
    )


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def maybe_freeze_stage1(model: nn.Module, epoch: int, cfg: Dict) -> None:
    """Freeze / unfreeze layer1 for ResUNet-18 per config."""
    if cfg["model"]["name"] != "resunet18":
        return
    freeze_until = cfg["model"].get("freeze_stage1_epochs", 0)
    should_freeze = epoch < freeze_until
    for p in model.layer1.parameters():
        p.requires_grad = not should_freeze
    if epoch == freeze_until and freeze_until > 0:
        logger.info("Epoch %d: unfreezing encoder layer1", epoch)


# ---------------------------------------------------------------------------
# Validation — sliding-window Dice
# ---------------------------------------------------------------------------

def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             model_name: str, loss_fn=None) -> Dict[str, float]:
    from monai.inferers import sliding_window_inference
    model.eval()
    dice_scores = {"wt": [], "tc": [], "et": []}
    val_losses = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            def _predict(x):
                if model_name == "vision_text":
                    seg, _, _, _ = model(x,
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device))
                    return seg
                return model(x)

            pred_logits = sliding_window_inference(
                image, roi_size=(96, 96, 96), sw_batch_size=1,
                predictor=_predict, overlap=0.5,
            )
            pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()

            for ch, key in enumerate(["wt", "tc", "et"]):
                p = pred_bin[:, ch]
                t = label[:, ch]
                inter = (p * t).sum()
                dice = (2 * inter / (p.sum() + t.sum() + 1e-8)).item()
                dice_scores[key].append(dice)

            # Val loss (same criterion as training)
            if loss_fn is not None:
                if model_name == "vision_text":
                    seg, slot_pred, img_emb, txt_emb = model(
                        image,
                        batch["input_ids"].to(device),
                        batch["attention_mask"].to(device),
                    )
                    vl, _ = loss_fn(seg, label, slot_pred,
                                    batch["slot_labels"].to(device),
                                    img_emb, txt_emb)
                else:
                    vl = loss_fn(pred_logits, label)
                val_losses.append(vl.item())

    means = {k: float(np.mean(v)) for k, v in dice_scores.items()}
    means["mean"] = float(np.mean(list(means.values())))
    means["val_loss"] = float(np.mean(val_losses)) if val_losses else 0.0
    return means


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler) -> int:
    logger.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt["epoch"]
    logger.info("Resuming from epoch %d", epoch)
    return epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scaler: Optional[GradScaler],
    device: torch.device,
    grad_accum: int,
    model_name: str,
    cfg: Dict,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        use_amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if model_name == "vision_text":
                seg_pred, slot_pred, img_emb, txt_emb = model(
                    image,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                slot_target = batch["slot_labels"].to(device)
                loss, _ = loss_fn(seg_pred, label, slot_pred, slot_target, img_emb, txt_emb)
            else:
                pred = model(image)
                loss = loss_fn(pred, label)

            loss = loss / grad_accum

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if scaler is not None and device.type == "cuda":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def log_csv(csv_path: str, row: Dict) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training script.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max_epochs from config (useful for CPU tests).")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Restrict training set to first N samples (CPU test mode).")
    parser.add_argument("--device", default=None,
                        help="Force device: 'cpu' or 'cuda'. Auto-detected if omitted.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI overrides
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Config: %s | Device: %s", args.config, device)
    set_seeds(cfg["training"].get("seed", 42))

    model_name = cfg["model"]["name"]
    save_dir    = cfg["checkpointing"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    csv_path   = os.path.join(save_dir, "train_log.csv")

    # ── datasets ──────────────────────────────────────────────────────────
    load_text = (model_name == "vision_text")
    tokenizer = None
    if load_text:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["biobert_name"])

    train_ds = BraTSDataset(
        split="train",
        splits_file=cfg["data"]["splits_file"],
        brats_root=cfg["data"]["brats_root"],
        textbrats_root=cfg["data"]["textbrats_root"],
        transforms=build_train_transforms(tuple(cfg["data"]["patch_size"])),
        load_text=load_text,
        tokenizer=tokenizer,
    )
    val_ds = BraTSDataset(
        split="val",
        splits_file=cfg["data"]["splits_file"],
        brats_root=cfg["data"]["brats_root"],
        textbrats_root=cfg["data"]["textbrats_root"],
        transforms=build_val_transforms(),
        load_text=load_text,
        tokenizer=tokenizer,
    )

    if args.num_samples is not None:
        logger.info("CPU test mode: restricting train set to %d samples", args.num_samples)
        train_ds = Subset(train_ds, list(range(min(args.num_samples, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(2, len(val_ds)))))

    num_workers = 0 if device.type == "cpu" else cfg["data"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── model / optimizer / scheduler / scaler ────────────────────────────
    model     = build_model(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler() if device.type == "cuda" and cfg["training"].get("mixed_precision") else None

    if model_name in ("resunet18", "swin_unetr"):
        loss_fn = SegmentationLoss()
    else:
        loss_fn = ModelCLoss(
            lambda_seg   = cfg.get("loss", {}).get("lambda_seg",   1.0),
            lambda_text  = cfg.get("loss", {}).get("lambda_text",  0.3),
            lambda_align = cfg.get("loss", {}).get("lambda_align", 0.1),
            slot_weights = tuple(cfg["model"].get("slot_weights", [1.0]*5)),
        )

    # ── auto-resume ───────────────────────────────────────────────────────
    last_ckpt = os.path.join(save_dir, "last.pth")
    start_epoch = 0
    best_dice   = 0.0
    if os.path.exists(last_ckpt):
        start_epoch = load_checkpoint(last_ckpt, model, optimizer, scheduler, scaler)
        best_ckpt = os.path.join(save_dir, "best.pth")
        if os.path.exists(best_ckpt):
            best_dice = torch.load(best_ckpt, map_location="cpu", weights_only=False).get("best_dice", 0.0)

    max_epochs = cfg["training"]["max_epochs"]
    grad_accum = cfg["training"].get("grad_accumulation_steps", 1)
    patience   = cfg["training"].get("early_stopping_patience", 12)
    no_improve = 0

    logger.info("Training %s for %d epochs (start=%d)", model_name, max_epochs, start_epoch)

    for epoch in range(start_epoch, max_epochs):
        maybe_freeze_stage1(model, epoch, cfg)

        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, device, grad_accum, model_name, cfg,
        )
        epoch_time = time.time() - t0

        val_dice = validate(model, val_loader, device, model_name, loss_fn)
        scheduler.step(val_dice["mean"])

        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val WT=%.4f TC=%.4f ET=%.4f mean=%.4f | %.1fs",
            epoch + 1, train_loss, val_dice["val_loss"],
            val_dice["wt"], val_dice["tc"], val_dice["et"], val_dice["mean"],
            epoch_time,
        )

        # CSV log
        log_csv(csv_path, {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_dice["val_loss"], 6),
            "val_wt":     round(val_dice["wt"], 6),
            "val_tc":     round(val_dice["tc"], 6),
            "val_et":     round(val_dice["et"], 6),
            "val_mean":   round(val_dice["mean"], 6),
            "epoch_secs": round(epoch_time, 1),
            "lr":         optimizer.param_groups[0]["lr"],
        })

        # Save last checkpoint every epoch
        state = {
            "epoch":      epoch + 1,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_dice":  best_dice,
        }
        if scaler:
            state["scaler"] = scaler.state_dict()
        save_checkpoint(state, last_ckpt)

        # Save best checkpoint
        if val_dice["mean"] > best_dice:
            best_dice = val_dice["mean"]
            state["best_dice"] = best_dice
            save_checkpoint(state, os.path.join(save_dir, "best.pth"))
            logger.info("  New best val Dice: %.4f — best.pth saved", best_dice)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch + 1, patience)
                break

    logger.info("Training complete. Best val Dice: %.4f", best_dice)
    logger.info("Checkpoints in: %s", save_dir)


if __name__ == "__main__":
    main()
