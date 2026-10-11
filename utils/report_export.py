"""Export analysis results: PNG images and a JSON report."""
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
from PIL import Image


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def build_report(filename: str, image_info: dict, detection=None, model_info: dict | None = None,
                 model_error: str | None = None, forensics: dict | None = None) -> dict:
    """Assemble a JSON-serialisable report containing only measured values."""
    report = {
        "report_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image": {"filename": filename, **image_info},
        "disclaimer": (
            "Scores are model outputs, not proof. A high score does not prove manipulation and a "
            "low score does not prove authenticity."
        ),
    }
    if detection is not None:
        report["deep_learning"] = {
            "method": (model_info or {}).get("arch", "ForensicUNet-ResNet18-SRM"),
            "checkpoint_epoch": (model_info or {}).get("epoch"),
            "trained_on": (model_info or {}).get("trained_on"),
            "assessment": detection.label,
            "is_suspicious": detection.is_suspicious,
            "image_score": round(detection.image_score, 4),
            "image_threshold": round(detection.image_threshold, 4),
            "pixel_threshold": round(detection.pixel_threshold, 3),
            "predicted_tampered_area_fraction": round(detection.tampered_fraction, 4),
            "inference_seconds": round(detection.inference_seconds, 3),
        }
    else:
        report["deep_learning"] = {"available": False, "reason": model_error}
    if forensics:
        report["classical_forensics"] = forensics
    return report


def report_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")
