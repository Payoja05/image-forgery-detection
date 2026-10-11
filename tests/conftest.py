import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def natural_rgb():
    """Deterministic textured RGB image (gradients + noise + shapes)."""
    rng = np.random.default_rng(0)
    h, w = 240, 320
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([x * 255 / w, y * 255 / h, (x + y) * 127 / (w + h)], -1)
    img += rng.normal(0, 6, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(img, (80, 80), 30, (200, 40, 40), -1)
    cv2.rectangle(img, (180, 120), (260, 200), (30, 160, 60), -1)
    return img


def encode(rgb, fmt="JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes(natural_rgb):
    return encode(natural_rgb, "JPEG", quality=90)
