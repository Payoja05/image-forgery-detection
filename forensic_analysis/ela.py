"""Error Level Analysis (ELA).

Idea: re-save the image as JPEG at a known quality and look at the absolute
difference. Regions that were already compressed at that quality change
little; regions with a different compression history (e.g. pasted from a
different JPEG, or edited after the last save) can change more.

Caveat: ELA also responds strongly to edges, texture, saturated colours and
resaving, so a bright region is a *hint*, never proof of tampering.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ELAResult:
    diff: np.ndarray          # H x W x 3 uint8 absolute difference (raw)
    visual: np.ndarray        # H x W x 3 uint8, brightness-scaled for viewing
    intensity: np.ndarray     # H x W float32 in [0, 1], per-pixel max-channel diff normalised
    quality: int
    max_diff: int
    mean_diff: float


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed during ELA.")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def error_level_analysis(rgb: np.ndarray, quality: int = 90, scale: float | None = None) -> ELAResult:
    """Compute ELA.

    scale: multiplier for the visualisation. None = auto-stretch so that the
    largest difference maps to 255 (the common "brightness" setting).
    """
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in 1..100")
    resaved = jpeg_roundtrip(rgb, quality)
    diff = cv2.absdiff(rgb, resaved)
    max_diff = int(diff.max())
    if scale is None:
        scale = 255.0 / max(max_diff, 1)
    visual = np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    per_pixel = diff.max(axis=2).astype(np.float32)
    intensity = per_pixel / max(float(per_pixel.max()), 1.0)
    return ELAResult(diff, visual, intensity, quality, max_diff, float(diff.mean()))


def ela_suspicion_map(rgb: np.ndarray, quality: int = 90, blur: int = 15) -> np.ndarray:
    """Smoothed ELA map in [0,1] used as a classical localisation baseline."""
    res = error_level_analysis(rgb, quality)
    k = blur | 1
    m = cv2.GaussianBlur(res.intensity, (k, k), 0)
    return m / max(float(m.max()), 1e-6)
