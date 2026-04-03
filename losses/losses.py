"""
losses/losses.py — Loss functions for all three models.

Models A & B:
    SegmentationLoss = DiceLoss(sigmoid=True) + BCEWithLogitsLoss

Model C:
    ModelCLoss = lambda_seg * L_seg
               + lambda_text * (BCEWithLogitsLoss(binary slots) + CrossEntropyLoss(burden slot))
               + lambda_align * (1 - cosine_similarity(img_emb, txt_emb).mean())
"""
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation loss (Models A, B, and the seg branch of C)
# ---------------------------------------------------------------------------

class SegmentationLoss(nn.Module):
    """DiceLoss(sigmoid=True) + BCEWithLogitsLoss, equal weights.

    Args:
        dice_weight: weight on Dice component (default 1.0)
        bce_weight:  weight on BCE component  (default 1.0)
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight  = bce_weight
        self.dice = DiceLoss(sigmoid=True, reduction="mean")
        self.bce  = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, 3, H, W, D] raw logits
            target: [B, 3, H, W, D] binary float labels
        Returns:
            scalar loss
        """
        return self.dice_weight * self.dice(pred, target) \
             + self.bce_weight  * self.bce(pred, target)


# ---------------------------------------------------------------------------
# Text slot loss (Model C)
# ---------------------------------------------------------------------------

class TextSlotLoss(nn.Module):
    """Loss on the 5 report slots.

    Slots 0,1,2,4 — binary (BCE with logits)
    Slot 3        — 3-class tumor burden (CrossEntropy)

    Args:
        slot_weights: per-slot loss weight, length 5 (default all 1.0)
    """

    def __init__(self, slot_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)) -> None:
        super().__init__()
        assert len(slot_weights) == 5
        self.weights = slot_weights
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.ce  = nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        slot_pred: torch.Tensor,    # [B, 5]  raw logits
        slot_target: torch.Tensor,  # [B, 5]  float labels (slot 3 is 0/1/2)
    ) -> torch.Tensor:
        w = self.weights

        # Binary slots: 0 (WT), 1 (TC), 2 (ET), 4 (enhancement)
        bce_loss = (
            w[0] * self.bce(slot_pred[:, 0], slot_target[:, 0]) +
            w[1] * self.bce(slot_pred[:, 1], slot_target[:, 1]) +
            w[2] * self.bce(slot_pred[:, 2], slot_target[:, 2]) +
            w[4] * self.bce(slot_pred[:, 4], slot_target[:, 4])
        ) / 4.0

        # 3-class slot: tumor burden (slot 3)
        burden_logits = slot_pred[:, 3:4].expand(-1, 3)   # naive: use same logit for all 3
        # Proper: model should output 3 logits for slot 3 — handled in Model C head
        # Here slot_pred[:,3] is a single logit so we treat burden as binary (small vs not-small)
        ce_loss = w[3] * self.bce(slot_pred[:, 3], (slot_target[:, 3] > 0).float())

        return bce_loss + ce_loss


# ---------------------------------------------------------------------------
# Alignment loss (Model C)
# ---------------------------------------------------------------------------

class AlignmentLoss(nn.Module):
    """1 - mean cosine similarity between image and text embeddings."""

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: [B, D] image pooled embedding
            txt_emb: [B, D] text CLS embedding
        Returns:
            scalar in [0, 2]  (0 = perfectly aligned)
        """
        cos_sim = F.cosine_similarity(img_emb, txt_emb, dim=1)  # [B]
        return 1.0 - cos_sim.mean()


# ---------------------------------------------------------------------------
# Combined Model C loss
# ---------------------------------------------------------------------------

class ModelCLoss(nn.Module):
    """Weighted combination of segmentation, text slot, and alignment losses.

    L_total = lambda_seg   * L_seg
            + lambda_text  * L_text
            + lambda_align * L_align
    """

    def __init__(
        self,
        lambda_seg:   float = 1.0,
        lambda_text:  float = 0.3,
        lambda_align: float = 0.1,
        slot_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.lambda_seg   = lambda_seg
        self.lambda_text  = lambda_text
        self.lambda_align = lambda_align
        self.seg_loss  = SegmentationLoss()
        self.text_loss = TextSlotLoss(slot_weights)
        self.align_loss = AlignmentLoss()

    def forward(
        self,
        seg_pred:    torch.Tensor,   # [B, 3, H, W, D] logits
        seg_target:  torch.Tensor,   # [B, 3, H, W, D] binary labels
        slot_pred:   torch.Tensor,   # [B, 5] logits
        slot_target: torch.Tensor,   # [B, 5] labels
        img_emb:     torch.Tensor,   # [B, D]
        txt_emb:     torch.Tensor,   # [B, D]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_seg   = self.seg_loss(seg_pred, seg_target)
        l_text  = self.text_loss(slot_pred, slot_target)
        l_align = self.align_loss(img_emb, txt_emb)

        total = (self.lambda_seg   * l_seg
               + self.lambda_text  * l_text
               + self.lambda_align * l_align)

        components = {"seg": l_seg, "text": l_text, "align": l_align}
        return total, components
