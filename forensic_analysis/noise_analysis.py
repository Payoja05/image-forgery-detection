"""Local noise analysis.

Every camera/processing chain leaves a characteristic noise level. When a
region is pasted from another image, blurred, or synthesised, its noise level
can differ from the rest of the picture.

Method:
 1. noise residual  r = gray - median3x3(gray)  (removes most image content)
 2. local noise level = std of r inside sliding blocks
 3. inconsistency = |local level - median level| / MAD  (robust z-score)

Caveat: smooth areas (sky, walls) naturally have low residual and textured
areas high residual, so differences in this map are not evidence on their own.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NoiseResult:
    residual_visual: np.ndarray     # H x W uint8, amplified |residual|
    noise_level: np.ndarray         # H x W float32, local std of residual
    inconsistency: np.ndarray       # H x W float32 in [0,1]
    global_noise: float
    block: int


def noise_residual(gray: np.ndarray) -> np.ndarray:
    return gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)


def local_std(x: np.ndarray, block: int) -> np.ndarray:
    k = (block, block)
    mean = cv2.blur(x, k)
    sq = cv2.blur(x * x, k)
    return np.sqrt(np.maximum(sq - mean * mean, 0))


def noise_analysis(rgb: np.ndarray, block: int = 16) -> NoiseResult:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r = noise_residual(gray)
    level = local_std(r, block)
    med = float(np.median(level))
    mad = float(np.median(np.abs(level - med))) + 1e-6
    z = np.abs(level - med) / (1.4826 * mad)
    inconsistency = np.clip(z / 6.0, 0, 1).astype(np.float32)   # z of 6 -> 1.0
    vis = np.clip(np.abs(r) * 12, 0, 255).astype(np.uint8)
    return NoiseResult(vis, level.astype(np.float32), inconsistency, med, block)
