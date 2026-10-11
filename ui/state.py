"""Session state + cached resources shared by all pages."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st

from config.settings import CHECKPOINT_PATH, DEFAULT_PIXEL_THRESHOLD, SAMPLES_DIR
from utils.image_validation import LoadedImage, load_path


@dataclass
class CurrentImage:
    loaded: LoadedImage
    source: str                          # "upload" | "sample"
    key: str                             # content hash -> ties results to this exact image
    ground_truth: np.ndarray | None = None   # bool mask, samples only
    sample_label: str | None = None
    sample_info: dict = field(default_factory=dict)


@st.cache_resource(show_spinner="Loading the forgery-localisation model ...")
def get_detector():
    """Load the model once per server process. Returns (detector, error message)."""
    from inference.forgery_detector import ForgeryDetector
    from models.model_loader import ModelLoadError

    try:
        return ForgeryDetector(CHECKPOINT_PATH), None
    except ModelLoadError as exc:
        return None, str(exc)
    except Exception as exc:  # unexpected - still show a clear message
        return None, f"Unexpected error while loading the model: {exc}"


def image_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def set_current(loaded: LoadedImage, source: str, **extra) -> None:
    key = image_key(loaded.raw_bytes)
    cur = st.session_state.get("current")
    if cur is not None and cur.key == key:
        return
    st.session_state["current"] = CurrentImage(loaded, source, key, **extra)
    st.session_state.pop("analysis", None)        # results belong to the previous image


def get_current() -> CurrentImage | None:
    return st.session_state.get("current")


def run_analysis(cur: CurrentImage) -> dict:
    """Run the real model on the current image and store the result."""
    detector, err = get_detector()
    if detector is None and Path(CHECKPOINT_PATH).exists():   # weights appeared after a failed load
        get_detector.clear()
        detector, err = get_detector()
    result = {"key": cur.key, "error": err, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if detector is not None:
        det = detector.detect(cur.loaded.rgb, DEFAULT_PIXEL_THRESHOLD)
        result.update({"prob_map": det.prob_map, "inference_seconds": det.inference_seconds,
                       "model_meta": detector.meta})
    st.session_state["analysis"] = result
    return result


def get_analysis(cur: CurrentImage | None) -> dict | None:
    a = st.session_state.get("analysis")
    if cur is None or a is None or a.get("key") != cur.key:
        return None
    return a


# ---------------------------------------------------------------- samples
def list_samples() -> list[dict]:
    meta = SAMPLES_DIR / "samples.json"
    if not meta.exists():
        return []
    items = json.loads(meta.read_text())
    return [s for s in items if (SAMPLES_DIR / s["file"]).exists()]


def load_sample(sample: dict) -> None:
    import cv2

    loaded = load_path(SAMPLES_DIR / sample["file"])
    gt = None
    if sample.get("mask"):
        m = cv2.imread(str(SAMPLES_DIR / sample["mask"]), cv2.IMREAD_GRAYSCALE)
        gt = m > 127 if m is not None else None
    set_current(loaded, "sample", ground_truth=gt, sample_label=sample["label"], sample_info=sample)


def model_available() -> bool:
    return Path(CHECKPOINT_PATH).exists()
