import json

import numpy as np
import pytest
import torch

from inference.forgery_detector import ForgeryDetector
from inference.localization import image_level_score, mask_from_probability, predict_probability_map
from models.model_loader import ModelLoadError, load_model, save_checkpoint
from models.unet import ForensicUNet
from utils.report_export import build_report, png_bytes, report_json_bytes
from visualization.overlays import blend_heatmap, heatmap, overlay_mask


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Randomly initialised (untrained) model saved in the real checkpoint format."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save_checkpoint(ForensicUNet(pretrained_encoder=False), path, {"image_threshold": 0.6, "arch_name": ForensicUNet.ARCH_NAME})
    return path


# ---------------------------------------------------------------- model loading
def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "nope.pt")


def test_corrupt_checkpoint(tmp_path):
    p = tmp_path / "bad.pt"
    p.write_bytes(b"garbage")
    with pytest.raises(ModelLoadError):
        load_model(p)


def test_wrong_architecture(tmp_path):
    p = tmp_path / "other.pt"
    torch.save({"arch": "SomethingElse", "state_dict": {}}, p)
    with pytest.raises(ModelLoadError, match="architecture"):
        load_model(p)


def test_load_ok(tiny_checkpoint):
    model, meta = load_model(tiny_checkpoint, torch.device("cpu"))
    assert not model.training and meta["image_threshold"] == 0.6


# ---------------------------------------------------------------- shapes
@pytest.mark.parametrize("hw", [(64, 64), (96, 160), (256, 256)])
def test_model_output_matches_input(hw):
    m = ForensicUNet().eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, *hw))
    assert out.shape == (2, 1, *hw)


def test_probability_map_at_original_resolution(tiny_checkpoint, natural_rgb):
    model, _ = load_model(tiny_checkpoint, torch.device("cpu"))
    big = np.tile(natural_rgb, (3, 3, 1))[:700, :950]          # not a multiple of 32, larger than 512
    prob = predict_probability_map(model, big)
    assert prob.shape == big.shape[:2] and prob.dtype == np.float32
    assert 0 <= prob.min() and prob.max() <= 1


def test_detector_end_to_end(tiny_checkpoint, natural_rgb):
    det = ForgeryDetector(tiny_checkpoint, torch.device("cpu"))
    r = det.detect(natural_rgb, pixel_threshold=0.5)
    assert r.mask.shape == natural_rgb.shape[:2] and r.image_threshold == 0.6
    assert r.is_suspicious == (r.image_score >= 0.6)
    assert r.inference_seconds > 0


# ---------------------------------------------------------------- masks & scores
def test_mask_from_probability_threshold_and_cleanup():
    prob = np.zeros((100, 100), np.float32)
    prob[20:60, 20:60] = 0.9          # real region
    prob[90, 90] = 0.99               # isolated speck -> removed
    m = mask_from_probability(prob, 0.5)
    assert set(np.unique(m)) == {0, 255}
    assert m[40, 40] == 255 and m[90, 90] == 0
    assert (mask_from_probability(prob, 0.95) == 0).all()


def test_image_level_score():
    prob = np.zeros((100, 100), np.float32)
    prob.flat[:100] = 1.0             # exactly top 1 %
    assert image_level_score(prob, 0.01) == pytest.approx(1.0)
    assert image_level_score(np.zeros((50, 50)), 0.01) == 0.0


# ---------------------------------------------------------------- overlays & export
def test_overlays(natural_rgb):
    mask = np.zeros(natural_rgb.shape[:2], np.uint8)
    mask[50:100, 50:100] = 255
    ov = overlay_mask(natural_rgb, mask)
    assert ov.shape == natural_rgb.shape
    assert np.array_equal(ov[0, 0], natural_rgb[0, 0]) and not np.array_equal(ov[75, 75], natural_rgb[75, 75])
    prob = np.random.default_rng(0).random(natural_rgb.shape[:2]).astype(np.float32)
    assert heatmap(prob).shape == natural_rgb.shape
    assert blend_heatmap(natural_rgb, prob).dtype == np.uint8


def test_json_report(tiny_checkpoint, natural_rgb):
    det = ForgeryDetector(tiny_checkpoint, torch.device("cpu")).detect(natural_rgb)
    rep = build_report("x.jpg", {"width": 320, "height": 240}, det, {"arch": "ForensicUNet-ResNet18-SRM", "epoch": 3})
    data = json.loads(report_json_bytes(rep))
    assert data["image"]["filename"] == "x.jpg"
    assert data["deep_learning"]["image_score"] == round(det.image_score, 4)
    assert "generated_at" in data and "disclaimer" in data
    no_model = build_report("y.png", {}, None, model_error="weights missing")
    assert no_model["deep_learning"] == {"available": False, "reason": "weights missing"}
    assert png_bytes(det.mask)[:8] == b"\x89PNG\r\n\x1a\n"
