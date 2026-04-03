"""
data/transforms.py — MONAI-compatible preprocessing and augmentation pipelines.

Pipeline order:
  1. ForegroundCropd       — crop to non-zero brain bounding box
  2. PerModalityNormalized — z-score per modality on non-zero voxels; zeros stay zero
  3. OptionalClipper       — clamp to [-5, 5]
  4. RemapLabelsd          — int seg [H,W,D] → float32 [3,H,W,D] (WT/TC/ET)
  5. (train only) RandSpatialCropd + augmentation
  6. ToTensord
"""
import logging
from typing import Dict, Hashable, Mapping, Tuple

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    MapTransform,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
)

from data.dataset import remap_labels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class ForegroundCropd(MapTransform):
    """Crop image and label to the bounding box of non-zero voxels in the image."""

    def __init__(self, keys: KeysCollection, image_key: str = "image") -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.image_key = image_key

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict:
        d = dict(data)
        image = d[self.image_key]  # [C, H, W, D]

        mask = np.any(image != 0, axis=0)  # [H, W, D]
        coords = np.where(mask)

        if len(coords[0]) == 0:
            logger.warning("ForegroundCropd: empty image, skipping crop.")
            return d

        mins = [int(c.min()) for c in coords]
        maxs = [int(c.max()) + 1 for c in coords]

        for key in self.keys:
            arr = d[key]
            if arr.ndim == 4:  # [C, H, W, D]
                d[key] = arr[:, mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
            elif arr.ndim == 3:  # [H, W, D]
                d[key] = arr[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
        return d


class PerModalityNormalized(MapTransform):
    """Z-score normalise each modality channel independently.

    Non-zero voxels define the mean/std. Zero voxels stay zero.
    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict:
        d = dict(data)
        for key in self.keys:
            image = d[key].astype(np.float32)  # [C, H, W, D]
            result = np.zeros_like(image)
            for c in range(image.shape[0]):
                channel = image[c]
                fg = channel != 0
                if fg.any():
                    mu = channel[fg].mean()
                    sigma = channel[fg].std()
                    if sigma > 1e-8:
                        result[c][fg] = (channel[fg] - mu) / sigma
                    else:
                        result[c][fg] = channel[fg] - mu
            d[key] = result
        return d


class OptionalClipper(MapTransform):
    """Clamp values to [clip_min, clip_max] for numerical stability."""

    def __init__(
        self,
        keys: KeysCollection,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
    ) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict:
        d = dict(data)
        for key in self.keys:
            d[key] = np.clip(d[key], self.clip_min, self.clip_max)
        return d


class RemapLabelsd(MapTransform):
    """Convert integer BraTS seg [H,W,D] → float32 binary channels [3,H,W,D]."""

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict:
        d = dict(data)
        for key in self.keys:
            d[key] = remap_labels(d[key].astype(np.int32))
        return d


class ToTensord(MapTransform):
    """Convert numpy arrays to torch float32 tensors."""

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict:
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            if isinstance(arr, np.ndarray):
                d[key] = torch.from_numpy(arr.copy())
        return d


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_train_transforms(patch_size: Tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """Training pipeline: crop → norm → clip → remap → random patch → augment → tensor."""
    keys = ["image", "label"]
    return Compose([
        ForegroundCropd(keys=keys, image_key="image"),
        PerModalityNormalized(keys=["image"]),
        OptionalClipper(keys=["image"], clip_min=-5.0, clip_max=5.0),
        RemapLabelsd(keys=["label"]),
        RandSpatialCropd(keys=keys, roi_size=patch_size, random_size=False),
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=2, prob=0.5),
        RandRotate90d(keys=keys, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.10),
        ToTensord(keys=keys),
    ])


def build_val_transforms() -> Compose:
    """Validation pipeline: crop → norm → clip → remap → tensor.

    No patch extraction — sliding-window inference handles that at eval time.
    """
    keys = ["image", "label"]
    return Compose([
        ForegroundCropd(keys=keys, image_key="image"),
        PerModalityNormalized(keys=["image"]),
        OptionalClipper(keys=["image"], clip_min=-5.0, clip_max=5.0),
        RemapLabelsd(keys=["label"]),
        ToTensord(keys=keys),
    ])
