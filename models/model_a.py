"""
models/model_a.py — 3D ResUNet-18

Architecture
------------
Encoder : 3D ResNet-18 stem + layer1..4  (MedicalNet pretrained, 1-ch → 4-ch adapted)
Decoder : 4 upsampling blocks with skip connections from each encoder stage
          Each block: ConvTranspose3d(up) → concat(skip) → Conv3d + BN + ReLU
Final   : ConvTranspose3d (×2 upsample, 48→96) → Conv3d(features, 3, 1)  [logits, no activation]

Spatial trace with input [B, 4, 96, 96, 96]:
  stem_conv  →  [B,  64, 48, 48, 48]   (stride 2, kernel 7)
  maxpool    →  [B,  64, 24, 24, 24]   (stride 2, kernel 3)
  layer1     →  [B,  64, 24, 24, 24]
  layer2     →  [B, 128, 12, 12, 12]   (stride 2)
  layer3     →  [B, 256,  6,  6,  6]   (stride 2)
  layer4     →  [B, 512,  3,  3,  3]   (stride 2)  ← bottleneck
  dec1 up    →  [B, 256,  6,  6,  6]  + layer3 skip
  dec2 up    →  [B, 128, 12, 12, 12]  + layer2 skip
  dec3 up    →  [B,  64, 24, 24, 24]  + layer1 skip
  dec4 up    →  [B,  32, 48, 48, 48]  + stem_conv skip
  final up   →  [B,  16, 96, 96, 96]
  head       →  [B,   3, 96, 96, 96]  logits
"""
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder building blocks
# ---------------------------------------------------------------------------

class BasicBlock3D(nn.Module):
    """3D ResNet-18 basic residual block."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(
    in_channels: int,
    out_channels: int,
    blocks: int,
    stride: int = 1,
) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels),
        )
    layers: List[nn.Module] = [BasicBlock3D(in_channels, out_channels, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(BasicBlock3D(out_channels, out_channels))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Decoder building block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """ConvTranspose3d upsample → concat skip → Conv3d + BN + ReLU."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResUNet-18
# ---------------------------------------------------------------------------

class ResUNet18(nn.Module):
    """3D ResUNet-18: MedicalNet ResNet-18 encoder + lightweight UNet decoder."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # stem: Conv(4,64,7,s2) + BN + ReLU  →  48³
        self.stem_conv = nn.Conv3d(in_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        self.stem_bn   = nn.BatchNorm3d(64)
        self.stem_relu = nn.ReLU(inplace=True)
        # maxpool  →  24³
        self.maxpool   = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(64,  64,  blocks=2, stride=1)   # 24³
        self.layer2 = _make_layer(64,  128, blocks=2, stride=2)   # 12³
        self.layer3 = _make_layer(128, 256, blocks=2, stride=2)   #  6³
        self.layer4 = _make_layer(256, 512, blocks=2, stride=2)   #  3³

        # ── Decoder ──────────────────────────────────────────────────────────
        # skip channels come from encoder stages
        self.dec1 = DecoderBlock(512, 256, 256)   # 3 → 6,   skip=layer3
        self.dec2 = DecoderBlock(256, 128, 128)   # 6 → 12,  skip=layer2
        self.dec3 = DecoderBlock(128,  64,  64)   # 12 → 24, skip=layer1
        self.dec4 = DecoderBlock( 64,  64,  32)   # 24 → 48, skip=stem_conv

        # final upsample 48 → 96, no skip
        self.final_up = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # output head — logits, no activation
        self.head = nn.Conv3d(16, out_channels, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self) -> None:
        for m in [self.dec1, self.dec2, self.dec3, self.dec4,
                  self.final_up, self.final_conv, self.head]:
            for module in (m.modules() if hasattr(m, 'modules') else [m]):
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(module, nn.BatchNorm3d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.stem_relu(self.stem_bn(self.stem_conv(x)))  # [B,64,48,48,48]
        s1 = self.layer1(self.maxpool(s0))                     # [B,64,24,24,24]
        s2 = self.layer2(s1)                                   # [B,128,12,12,12]
        s3 = self.layer3(s2)                                   # [B,256,6,6,6]
        bottleneck = self.layer4(s3)                           # [B,512,3,3,3]

        # Decoder
        d = self.dec1(bottleneck, s3)   # [B,256,6,6,6]
        d = self.dec2(d, s2)            # [B,128,12,12,12]
        d = self.dec3(d, s1)            # [B,64,24,24,24]
        d = self.dec4(d, s0)            # [B,32,48,48,48]

        d = self.final_conv(self.final_up(d))  # [B,16,96,96,96]
        return self.head(d)                    # [B,3,96,96,96]


# ---------------------------------------------------------------------------
# MedicalNet weight loading
# ---------------------------------------------------------------------------

def load_medicalnet_weights(model: ResUNet18, weights_path: str) -> None:
    """Load MedicalNet ResNet-18 pretrained weights into the encoder.

    Handles:
    - 'module.' prefix removal (DataParallel checkpoint)
    - 1-channel → 4-channel conv1 adaptation (mean + tile, scaled by 1/4)
    - Decoder weights are left as randomly initialised
    - Mismatched / missing keys are skipped and logged
    """
    logger.info("Loading MedicalNet weights from: %s", weights_path)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    src_sd = checkpoint["state_dict"]

    # Strip 'module.' prefix
    src_sd = {k.replace("module.", ""): v for k, v in src_sd.items()}

    # Remap MedicalNet key names to our model's attribute names.
    # MedicalNet uses conv1/bn1 for the stem; we use stem_conv/stem_bn.
    remap = {
        "conv1.weight":         "stem_conv.weight",
        "bn1.weight":           "stem_bn.weight",
        "bn1.bias":             "stem_bn.bias",
        "bn1.running_mean":     "stem_bn.running_mean",
        "bn1.running_var":      "stem_bn.running_var",
        "bn1.num_batches_tracked": "stem_bn.num_batches_tracked",
    }
    src_sd = {remap.get(k, k): v for k, v in src_sd.items()}

    model_sd = model.state_dict()
    to_load = {}
    skipped = []

    for src_key, src_val in src_sd.items():
        if src_key not in model_sd:
            skipped.append(f"not-in-model: {src_key}")
            continue

        model_val = model_sd[src_key]

        # Adapt stem_conv: [64,1,7,7,7] → [64,4,7,7,7]
        if src_key == "stem_conv.weight" and src_val.shape != model_val.shape:
            # Average 1-channel weight, replicate 4 times, scale to keep
            # similar activation magnitude (each of the 4 channels contributes 1/4)
            avg = src_val.mean(dim=1, keepdim=True)   # [64,1,7,7,7]
            adapted = avg.repeat(1, 4, 1, 1, 1) / 4.0  # [64,4,7,7,7]
            to_load[src_key] = adapted
            logger.info("  conv1 adapted: %s → %s", src_val.shape, adapted.shape)
            continue

        if src_val.shape != model_val.shape:
            skipped.append(f"shape-mismatch: {src_key} {src_val.shape} vs {model_val.shape}")
            continue

        to_load[src_key] = src_val

    model_sd.update(to_load)
    model.load_state_dict(model_sd)

    logger.info("  Loaded  : %d keys", len(to_load))
    logger.info("  Skipped : %d keys", len(skipped))
    for s in skipped:
        logger.debug("    skip: %s", s)
    logger.info("MedicalNet weights loaded successfully.")
