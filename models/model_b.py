"""
models/model_b.py — Swin UNETR wrapper for BraTS 3D segmentation

Architecture
------------
MONAI SwinUNETR with:
  - in_channels=4  (T1, T1ce, T2, FLAIR)
  - out_channels=3 (WT, TC, ET logits — no activation)
  - feature_size=48
  - use_checkpoint=True  (gradient checkpointing for Kaggle VRAM)

Pretrained weights from BraTS self-supervised pretraining (SSL).
Partial loading: mismatched or extra keys are skipped with a logged warning.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SwinUNETRWrapper(nn.Module):
    """Thin wrapper around MONAI SwinUNETR.

    Handles:
    - Partial pretrained weight loading (SSL checkpoint has different head)
    - Unified forward() returning raw logits [B, 3, H, W, D]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        pretrained_weights: Optional[str] = None,
    ) -> None:
        super().__init__()

        from monai.networks.nets import SwinUNETR

        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        if pretrained_weights is not None:
            self._load_pretrained(pretrained_weights)

    # ------------------------------------------------------------------
    def _load_pretrained(self, weights_path: str) -> None:
        """Load SSL pretrained weights with partial matching.

        The BraTS SSL checkpoint was trained with a reconstruction head
        that does not exist in the fine-tuning model, so those keys are
        skipped. All backbone (encoder) keys that match in name and shape
        are loaded.
        """
        logger.info("Loading Swin UNETR pretrained weights from: %s", weights_path)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        # Different checkpoints use different top-level keys
        if "state_dict" in checkpoint:
            src_sd = checkpoint["state_dict"]
        elif "model" in checkpoint:
            src_sd = checkpoint["model"]
        else:
            src_sd = checkpoint

        # The BraTS SSL checkpoint was saved with DataParallel:
        #   checkpoint key : "module.patch_embed.proj.weight"
        #   MONAI model key: "swinViT.patch_embed.proj.weight"
        # Strategy: strip "module." and prepend "swinViT." for the ViT backbone.
        # SSL-only keys (rotation_head, contrastive_head, convTrans3d) will
        # not match and are safely skipped.
        model_sd = self.swin.state_dict()

        loaded, skipped_shape, skipped_missing = [], [], []

        for src_key, src_val in src_sd.items():
            bare = src_key.replace("module.", "", 1)   # strip DataParallel prefix

            # Try several candidate remappings in priority order
            candidates = [
                bare,                     # no prefix change
                "swinViT." + bare,        # SSL backbone → MONAI swinViT block
            ]

            matched_key = None
            for cand in candidates:
                if cand in model_sd:
                    matched_key = cand
                    break

            if matched_key is None:
                skipped_missing.append(src_key)
                continue

            if src_val.shape != model_sd[matched_key].shape:
                # patch_embed: [48,1,2,2,2] → [48,4,2,2,2] (1-ch SSL → 4-ch MRI)
                # Same mean-tile strategy as MedicalNet conv1 adaptation.
                model_val = model_sd[matched_key]
                if (src_val.dim() == model_val.dim()
                        and src_val.shape[0] == model_val.shape[0]
                        and src_val.shape[1] == 1
                        and model_val.shape[1] == 4):
                    avg = src_val.mean(dim=1, keepdim=True)       # [C,1,...]
                    adapted = avg.repeat(1, 4, *([1] * (src_val.dim() - 2))) / 4.0
                    model_sd[matched_key] = adapted
                    loaded.append(matched_key)
                    logger.info("  patch_embed adapted: %s → %s",
                                list(src_val.shape), list(adapted.shape))
                    continue
                skipped_shape.append(
                    f"{src_key}: src={list(src_val.shape)} vs model={list(model_sd[matched_key].shape)}"
                )
                continue

            model_sd[matched_key] = src_val
            loaded.append(matched_key)

        self.swin.load_state_dict(model_sd)

        logger.info("  Loaded  : %d / %d keys", len(loaded), len(src_sd))
        logger.info("  Skipped (not in model) : %d", len(skipped_missing))
        logger.info("  Skipped (shape mismatch): %d", len(skipped_shape))
        for s in skipped_shape:
            logger.debug("    shape-skip: %s", s)
        logger.info("Swin UNETR pretrained weights loaded.")

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 4, H, W, D] multi-modal MRI patch (normalised, clipped)

        Returns:
            [B, 3, H, W, D] logits for WT, TC, ET (no sigmoid)
        """
        return self.swin(x)
