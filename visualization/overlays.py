"""Heatmaps, mask overlays and simple colour maps for display/export."""
from __future__ import annotations

import cv2
import numpy as np


def heatmap(prob: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """[0,1] float map -> RGB heatmap (uint8). Absolute scale: 0 = blue, 1 = red,
    so heatmaps of different images are comparable (no per-image stretching)."""
    u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, colormap), cv2.COLOR_BGR2RGB)


def blend_heatmap(rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(rgb, 1 - alpha, heatmap(prob), alpha, 0)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(220, 50, 47), alpha: float = 0.4,
                 outline: bool = True) -> np.ndarray:
    """Tint mask>0 pixels with `color` and draw their outline."""
    out = rgb.copy()
    m = mask > 0
    if m.any():
        tint = np.empty_like(rgb)
        tint[:] = color
        out[m] = (rgb[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
        if outline:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, max(2, rgb.shape[1] // 250))
    return out


def gray_to_rgb(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)


def colorize(x: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    return heatmap(x, colormap)
