"""Validation of user-supplied image files before any processing happens."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import (
    ALLOWED_EXTENSIONS,
    ALLOWED_PIL_FORMATS,
    MAX_FILE_SIZE_MB,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_SIDE,
)

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be used for analysis."""


@dataclass
class LoadedImage:
    name: str
    rgb: np.ndarray            # H x W x 3, uint8, RGB order
    format: str                # "JPEG" / "PNG"
    size_bytes: int
    raw_bytes: bytes           # untouched original bytes (needed for ELA etc.)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024


def validate_and_load(data: bytes, filename: str) -> LoadedImage:
    """Check extension, size, real file format and dimensions; return RGB array.

    Raises ImageValidationError with a human-readable message on failure.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{ext or 'none'}'. Please upload a JPG, JPEG or PNG image."
        )
    if len(data) == 0:
        raise ImageValidationError("The file is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ImageValidationError(f"File is larger than {MAX_FILE_SIZE_MB} MB.")

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                      # detects truncated / corrupt files
        img = Image.open(io.BytesIO(data))  # verify() invalidates the object
        fmt = img.format
        img = ImageOps.exif_transpose(img)  # respect camera orientation
        rgb = np.array(img.convert("RGB"))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("Image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("The file could not be read as an image (corrupt or not an image).") from exc

    if fmt not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(f"File content is {fmt}, not JPEG/PNG.")
    h, w = rgb.shape[:2]
    if min(h, w) < MIN_IMAGE_SIDE:
        raise ImageValidationError(f"Image is too small ({w}x{h}); minimum side is {MIN_IMAGE_SIDE}px.")

    return LoadedImage(name=filename, rgb=rgb, format=fmt, size_bytes=len(data), raw_bytes=data)


def load_path(path: str | Path) -> LoadedImage:
    path = Path(path)
    return validate_and_load(path.read_bytes(), path.name)
