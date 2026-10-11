"""Edge and sharpness analysis with OpenCV.

Spliced objects sometimes have unnaturally sharp or unnaturally soft
boundaries, and a pasted region may be sharper/blurrier than its surroundings.
We show (a) Canny edges and (b) a block-wise sharpness map (variance of the
Laplacian). Both are visual aids for a human analyst - natural depth-of-field
and textures produce the same patterns, so they are not detectors by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeResult:
    edges: np.ndarray        # H x W uint8 (0/255) Canny edge map
    sharpness: np.ndarray    # H x W float32 in [0,1], block-wise Laplacian variance (log-scaled)
    edge_density: float      # fraction of edge pixels


def edge_analysis(rgb: np.ndarray, low: int = 60, high: int = 160, block: int = 24) -> EdgeResult:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, low, high)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    k = (block, block)
    mean = cv2.blur(lap, k)
    var = np.maximum(cv2.blur(lap * lap, k) - mean * mean, 0)
    s = np.log1p(var)
    s = (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
    return EdgeResult(edges, s.astype(np.float32), float((edges > 0).mean()))
