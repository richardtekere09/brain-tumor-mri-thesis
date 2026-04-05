"""
models/model_c.py — Vision-Text model for BraTS 3D segmentation + report generation

Architecture
------------
Image branch : SwinUNETR (same as Model B) — returns seg logits [B,3,H,W,D]
               Bottleneck (encoder10) → global avg pool → image embedding [B,768]
Text branch  : BioBERT (dmis-lab/biobert-v1.1, frozen) — CLS token [B,768]
Fusion       : Gated cross-modal fusion → fused embedding [B,768]
Slot head    : Linear(768, 5) — report-slot logits [B,5]

Forward returns: seg_logits, slot_logits, img_emb, txt_emb

Tasks covered
-------------
4.1 — BioBERTEncoder
4.2 — FusionBlock
4.3 — ReportSlotHead
4.4 — VisionTextModel (full forward)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4.1 — BioBERT text encoder
# ─────────────────────────────────────────────────────────────────────────────

class BioBERTEncoder(nn.Module):
    """Frozen BioBERT encoder — returns CLS-token embedding [B, 768].

    Args:
        model_name: HuggingFace model ID (default: 'dmis-lab/biobert-v1.1').
        freeze:     If True (default), no gradients flow through this module.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-v1.1",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        logger.info(
            "BioBERTEncoder: loaded '%s'  frozen=%s  hidden_size=%d",
            model_name, freeze, self.bert.config.hidden_size,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]

        Returns:
            [B, 768] — CLS token embedding (last hidden state, position 0)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]   # CLS token


# ─────────────────────────────────────────────────────────────────────────────
# Task 4.2 — Gated cross-modal fusion block
# ─────────────────────────────────────────────────────────────────────────────

class FusionBlock(nn.Module):
    """Gated fusion of image and text embeddings.

    The gate is a learned sigmoid function conditioned on both modalities.
    It interpolates between the projected image and text representations:

        gate  = sigmoid(W [img; txt])
        fused = gate * proj(img) + (1 - gate) * proj(txt)
        out   = LayerNorm(fused)

    Args:
        image_dim:  Dimension of image embedding (default 768).
        text_dim:   Dimension of text embedding (default 768).
        output_dim: Output dimension (default 768).
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        output_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_proj  = nn.Linear(image_dim, output_dim)
        self.txt_proj  = nn.Linear(text_dim, output_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(image_dim + text_dim, output_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: [B, image_dim]
            txt_emb: [B, text_dim]

        Returns:
            [B, output_dim] — fused embedding
        """
        gate   = self.gate_layer(torch.cat([img_emb, txt_emb], dim=-1))
        img_f  = self.img_proj(img_emb)
        txt_f  = self.txt_proj(txt_emb)
        fused  = gate * img_f + (1.0 - gate) * txt_f
        return self.norm(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4.3 — Report-slot head
# ─────────────────────────────────────────────────────────────────────────────

class ReportSlotHead(nn.Module):
    """Linear projection from fused embedding to report-slot logits.

    5 output slots (all logits, no activation — apply sigmoid/softmax at loss):
        0 — WT present     (binary)
        1 — TC present     (binary)
        2 — ET present     (binary)
        3 — tumor burden   (3-class: small=0, medium=1, large=2)
        4 — enhancement    (binary: limited=0, prominent=1)

    Args:
        input_dim: Dimension of the fused embedding (default 768).
        num_slots: Number of output slots (default 5).
    """

    def __init__(self, input_dim: int = 768, num_slots: int = 5) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, num_slots)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: [B, input_dim]

        Returns:
            [B, num_slots] — raw logits
        """
        return self.head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4.4 — Full Vision-Text model
# ─────────────────────────────────────────────────────────────────────────────

class VisionTextModel(nn.Module):
    """Combined vision-text model for 3D brain tumor segmentation + report slots.

    Image branch:
        - SwinUNETR backbone (same weights/config as Model B)
        - encoder10 bottleneck [B, 768, 6, 6, 6] → global avg pool → [B, 768]
        - Linear projection → img_emb [B, 768]

    Text branch:
        - BioBERT (frozen) → txt_emb [B, 768]

    Fusion + slots:
        - FusionBlock(img_emb, txt_emb) → fused [B, 768]
        - ReportSlotHead(fused) → slot_logits [B, 5]

    Forward returns:
        (seg_logits [B,3,H,W,D], slot_logits [B,5], img_emb [B,768], txt_emb [B,768])

    Args:
        in_channels:    Number of MRI modalities (default 4).
        out_channels:   Number of segmentation classes (default 3).
        feature_size:   SwinUNETR feature size (default 48).
                        Bottleneck channels = 16 * feature_size = 768.
        swin_weights:   Optional path to SSL pretrained SwinUNETR weights.
        biobert_name:   HuggingFace model ID for BioBERT.
        freeze_biobert: Whether to freeze BioBERT (default True).
        fusion_dim:     Dimension for fusion and slot head (default 768).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 48,
        swin_weights: Optional[str] = None,
        biobert_name: str = "dmis-lab/biobert-v1.1",
        freeze_biobert: bool = True,
        fusion_dim: int = 768,
    ) -> None:
        super().__init__()

        # ── Image branch (SwinUNETR) ─────────────────────────────────────────
        from models.model_b import SwinUNETRWrapper
        self.swin_wrapper = SwinUNETRWrapper(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            pretrained_weights=swin_weights,
        )

        # Hook encoder10 to capture bottleneck features [B, 16*feature_size, 6, 6, 6]
        self._enc10_feat: Optional[torch.Tensor] = None
        self.swin_wrapper.swin.encoder10.register_forward_hook(
            lambda m, i, o: setattr(self, "_enc10_feat", o)
        )

        # Project bottleneck channels (16*feature_size) → fusion_dim
        bottleneck_ch = 16 * feature_size   # 768 for feature_size=48
        self.img_proj = nn.Linear(bottleneck_ch, fusion_dim)

        # ── Text branch (BioBERT) ─────────────────────────────────────────────
        self.biobert = BioBERTEncoder(model_name=biobert_name, freeze=freeze_biobert)

        # ── Fusion + slot head ────────────────────────────────────────────────
        self.fusion    = FusionBlock(image_dim=fusion_dim, text_dim=768, output_dim=fusion_dim)
        self.slot_head = ReportSlotHead(input_dim=fusion_dim, num_slots=5)

        logger.info(
            "VisionTextModel: feature_size=%d  bottleneck_ch=%d  fusion_dim=%d",
            feature_size, bottleneck_ch, fusion_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image:          [B, 4, H, W, D] — normalised multi-modal MRI patch
            input_ids:      [B, seq_len]    — tokenised text report
            attention_mask: [B, seq_len]

        Returns:
            seg_logits:  [B, 3, H, W, D] — segmentation logits (WT, TC, ET)
            slot_logits: [B, 5]          — report-slot logits
            img_emb:     [B, 768]        — projected image embedding
            txt_emb:     [B, 768]        — BioBERT CLS embedding
        """
        # Image branch — hook fills self._enc10_feat during this call
        seg_logits = self.swin_wrapper(image)   # [B, 3, H, W, D]

        # Global-average-pool bottleneck features → [B, bottleneck_ch]
        enc10 = self._enc10_feat                # [B, bottleneck_ch, d, h, w]
        img_pool = enc10.mean(dim=(2, 3, 4))    # [B, bottleneck_ch]
        img_emb  = self.img_proj(img_pool)      # [B, fusion_dim]

        # Text branch
        txt_emb = self.biobert(input_ids, attention_mask)   # [B, 768]

        # Fusion → slot prediction
        fused       = self.fusion(img_emb, txt_emb)         # [B, fusion_dim]
        slot_logits = self.slot_head(fused)                 # [B, 5]

        return seg_logits, slot_logits, img_emb, txt_emb
