import numpy as np
import pytest
from conftest import encode

from preprocessing.image_preprocessing import image_quality_report, normalize, pad_to_multiple, prepare_for_model, resize_max_side
from utils.image_validation import ImageValidationError, validate_and_load


# ---------------------------------------------------------------- validation
def test_valid_jpeg_upload(jpeg_bytes):
    img = validate_and_load(jpeg_bytes, "photo.jpg")
    assert img.format == "JPEG" and img.rgb.shape == (240, 320, 3) and img.rgb.dtype == np.uint8


def test_valid_png_upload(natural_rgb):
    img = validate_and_load(encode(natural_rgb, "PNG"), "photo.PNG")
    assert img.format == "PNG"
    assert np.array_equal(img.rgb, natural_rgb)          # lossless round-trip


@pytest.mark.parametrize("name", ["doc.pdf", "image.gif", "noext"])
def test_rejects_bad_extension(jpeg_bytes, name):
    with pytest.raises(ImageValidationError, match="Unsupported"):
        validate_and_load(jpeg_bytes, name)


def test_rejects_non_image_content():
    with pytest.raises(ImageValidationError, match="could not be read"):
        validate_and_load(b"this is not an image at all" * 20, "fake.jpg")


def test_rejects_truncated_file(jpeg_bytes):
    with pytest.raises(ImageValidationError):
        validate_and_load(jpeg_bytes[: len(jpeg_bytes) // 3], "broken.jpg")


def test_rejects_empty_and_tiny(natural_rgb):
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_load(b"", "a.jpg")
    with pytest.raises(ImageValidationError, match="too small"):
        validate_and_load(encode(natural_rgb[:20, :20], "PNG"), "tiny.png")


def test_rejects_mismatched_content(natural_rgb):
    with pytest.raises(ImageValidationError, match="not JPEG/PNG"):
        validate_and_load(encode(natural_rgb, "BMP"), "sneaky.png")


# ---------------------------------------------------------------- preprocessing
def test_resize_never_upscales(natural_rgb):
    out, s = resize_max_side(natural_rgb, 1000)
    assert s == 1.0 and out.shape == natural_rgb.shape
    out, s = resize_max_side(natural_rgb, 160)
    assert max(out.shape[:2]) == 160 and s == pytest.approx(0.5)


def test_pad_to_multiple(natural_rgb):
    padded, hw = pad_to_multiple(natural_rgb[:100, :150], 32)
    assert padded.shape[:2] == (128, 160) and hw == (100, 150)


def test_normalize_shape_and_stats(natural_rgb):
    x = normalize(natural_rgb)
    assert x.shape == (3, 240, 320) and x.dtype == np.float32
    assert -3 < x.mean() < 3


def test_prepare_for_model_dimensions(natural_rgb):
    big = np.tile(natural_rgb, (3, 3, 1))             # 720 x 960
    inp = prepare_for_model(big, max_side=512)
    assert inp.tensor.shape[0] == 1 and inp.tensor.shape[1] == 3
    assert inp.tensor.shape[2] % 32 == 0 and inp.tensor.shape[3] % 32 == 0
    assert inp.working_hw == (384, 512) and inp.original_hw == (720, 960)


def test_quality_report_flags(natural_rgb):
    r = image_quality_report(natural_rgb)
    assert r["width"] == 320 and not r["is_near_uniform"]
    flat = np.full((300, 300, 3), 128, np.uint8)
    assert image_quality_report(flat)["is_near_uniform"]
