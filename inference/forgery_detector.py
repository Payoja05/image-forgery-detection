"""High-level detector used by the app and the evaluation script."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from config.settings import (
    CHECKPOINT_PATH,
    DEFAULT_IMAGE_THRESHOLD,
    DEFAULT_PIXEL_THRESHOLD,
    IMAGE_SCORE_TOP_FRACTION,
)
from inference.localization import image_level_score, mask_from_probability, predict_probability_map
from models.model_loader import load_model

SUSPICIOUS = "Suspicious — signs of manipulation"
NOT_DETECTED = "No manipulation detected by the model"


@dataclass
class DetectionResult:
    prob_map: np.ndarray        # H x W float32 [0,1]
    mask: np.ndarray            # H x W uint8 0/255 at the chosen pixel threshold
    image_score: float          # [0,1]
    image_threshold: float
    pixel_threshold: float
    label: str
    is_suspicious: bool
    tampered_fraction: float
    inference_seconds: float


class ForgeryDetector:
    def __init__(self, checkpoint=CHECKPOINT_PATH, device=None):
        self.model, self.meta = load_model(checkpoint, device)
        self.image_threshold = float(self.meta.get("image_threshold", DEFAULT_IMAGE_THRESHOLD))

    @property
    def name(self) -> str:
        return self.meta.get("arch_name", "ForensicUNet-ResNet18-SRM")

    def detect(self, rgb: np.ndarray, pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD) -> DetectionResult:
        t0 = time.perf_counter()
        prob = predict_probability_map(self.model, rgb)
        elapsed = time.perf_counter() - t0
        return self.result_from_prob(prob, pixel_threshold, elapsed)

    def result_from_prob(self, prob: np.ndarray, pixel_threshold: float, elapsed: float = 0.0) -> DetectionResult:
        score = image_level_score(prob, IMAGE_SCORE_TOP_FRACTION)
        mask = mask_from_probability(prob, pixel_threshold)
        suspicious = score >= self.image_threshold
        return DetectionResult(
            prob_map=prob,
            mask=mask,
            image_score=score,
            image_threshold=self.image_threshold,
            pixel_threshold=pixel_threshold,
            label=SUSPICIOUS if suspicious else NOT_DETECTED,
            is_suspicious=bool(suspicious),
            tampered_fraction=float((mask > 0).mean()),
            inference_seconds=elapsed,
        )
