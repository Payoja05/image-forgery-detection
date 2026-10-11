"""Image preprocessing shared by training, evaluation and the app.

The original image is never modified in place; every function returns a new
array so that the untouched original stays available for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config.settings import IMAGENET_MEAN, IMAGENET_STD, INFER_MAX_SIDE, STRIDE_MULTIPLE


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_max_side(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Down-scale (never up-scale) so that max(H, W) <= max_side.

    INTER_AREA is used because it averages pixels instead of skipping them,
    which keeps more of the local noise statistics the model relies on.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return rgb.copy(), 1.0
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(rgb, new, interpolation=cv2.INTER_AREA), scale


def pad_to_multiple(rgb: np.ndarray, multiple: int = STRIDE_MULTIPLE) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad bottom/right so H and W are multiples of `multiple`."""
    h, w = rgb.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return rgb, (h, w)
    return cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_REFLECT_101), (h, w)


def normalize(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB (H,W,3) -> float32 CHW normalised with ImageNet statistics."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@dataclass
class ModelInput:
    tensor: np.ndarray              # 1 x 3 x Hp x Wp float32
    working_hw: tuple[int, int]     # size after resizing, before padding
    original_hw: tuple[int, int]
    scale: float


def prepare_for_model(rgb: np.ndarray, max_side: int = INFER_MAX_SIDE) -> ModelInput:
    """Full inference preprocessing: resize -> pad -> normalise -> batch dim."""
    original_hw = rgb.shape[:2]
    resized, scale = resize_max_side(rgb, max_side)
    padded, working_hw = pad_to_multiple(resized)
    return ModelInput(normalize(padded)[None], working_hw, original_hw, scale)


def image_quality_report(rgb: np.ndarray) -> dict:
    """Cheap quality indicators that affect how much the analysis can be trusted."""
    gray = to_gray(rgb)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean_brightness": round(float(gray.mean()), 1),
        "contrast_std": round(float(gray.std()), 1),
        "sharpness_laplacian_var": round(sharpness, 1),
        "is_low_resolution": bool(min(rgb.shape[:2]) < 256),
        "is_very_blurry": bool(sharpness < 50),
        "is_near_uniform": bool(gray.std() < 8),
    }
