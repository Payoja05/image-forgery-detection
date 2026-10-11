"""Model output -> probability map -> binary tampering mask."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from config.settings import INFER_MAX_SIDE
from preprocessing.image_preprocessing import prepare_for_model


@torch.no_grad()
def predict_probability_map(model: torch.nn.Module, rgb: np.ndarray, max_side: int = INFER_MAX_SIDE) -> np.ndarray:
    """Run the network and return per-pixel tampering probability at the
    *original* image resolution (float32, H x W, values in [0, 1])."""
    inp = prepare_for_model(rgb, max_side)
    device = next(model.parameters()).device
    x = torch.from_numpy(inp.tensor).to(device)
    logits = model(x)[0, 0]
    h, w = inp.working_hw
    prob = torch.sigmoid(logits[:h, :w]).float().cpu().numpy()        # drop padding
    oh, ow = inp.original_hw
    if (h, w) != (oh, ow):
        prob = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def mask_from_probability(prob: np.ndarray, threshold: float = 0.5, min_area_fraction: float = 0.001) -> np.ndarray:
    """Threshold + light post-processing.

    1. binary = prob >= threshold
    2. morphological opening (3x3) removes isolated speckle pixels
    3. connected components smaller than `min_area_fraction` of the image are
       dropped (they are almost always noise at this model's resolution)
    Returns uint8 mask with values 0 / 255.
    """
    mask = (prob >= threshold).astype(np.uint8)
    if mask.any():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = min_area_fraction * mask.size
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels].astype(np.uint8)
    return mask * 255


def image_level_score(prob: np.ndarray, top_fraction: float = 0.01) -> float:
    """Image-level manipulation score = mean of the top `top_fraction` pixel
    probabilities. Using a small top-fraction (instead of the max) makes the
    score robust to single noisy pixels while still reacting to small regions."""
    flat = prob.ravel()
    k = max(1, int(len(flat) * top_fraction))
    return float(np.partition(flat, -k)[-k:].mean())
