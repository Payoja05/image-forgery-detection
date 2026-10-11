import cv2
import numpy as np

from forensic_analysis.copy_move import detect_copy_move
from forensic_analysis.edge_analysis import edge_analysis
from forensic_analysis.ela import ela_suspicion_map, error_level_analysis, jpeg_roundtrip
from forensic_analysis.noise_analysis import noise_analysis


def test_ela_output(natural_rgb):
    r = error_level_analysis(natural_rgb, quality=90)
    assert r.diff.shape == natural_rgb.shape and r.visual.dtype == np.uint8
    assert r.intensity.shape == natural_rgb.shape[:2]
    assert 0 <= r.intensity.min() and r.intensity.max() <= 1
    assert r.max_diff > 0


def test_ela_highlights_region_with_different_compression(natural_rgb):
    """Paste an uncompressed patch into a heavily compressed image: ELA should be higher there."""
    base = jpeg_roundtrip(jpeg_roundtrip(natural_rgb, 70), 70)
    forged = base.copy()
    forged[60:140, 100:220] = natural_rgb[60:140, 100:220]     # never compressed
    m = ela_suspicion_map(forged, quality=70)
    assert m[60:140, 100:220].mean() > 1.5 * np.delete(m, np.s_[60:140], 0).mean()


def test_noise_analysis_detects_noise_mismatch():
    rng = np.random.default_rng(1)
    img = np.clip(128 + rng.normal(0, 2, (200, 200, 3)), 0, 255).astype(np.uint8)
    img[60:140, 60:140] = np.clip(128 + rng.normal(0, 15, (80, 80, 3)), 0, 255).astype(np.uint8)
    r = noise_analysis(img, block=16)
    assert r.inconsistency[80:120, 80:120].mean() > r.inconsistency[:40, :40].mean() + 0.3


def test_edge_analysis(natural_rgb):
    r = edge_analysis(natural_rgb)
    assert r.edges.shape == natural_rgb.shape[:2] and set(np.unique(r.edges)) <= {0, 255}
    assert 0 < r.edge_density < 1 and r.sharpness.max() <= 1


def test_copy_move_detects_cloned_patch():
    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (3, 3), 0)
    img[200:270, 300:380] = img[30:100, 40:120]                # clone a textured patch
    r = detect_copy_move(img)
    assert r.detected and r.num_inliers >= 8
    assert r.mask[230, 340] == 255 and r.mask[60, 80] == 255


def test_copy_move_no_false_alarm_on_random_texture():
    rng = np.random.default_rng(4)
    img = cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (3, 3), 0)
    assert not detect_copy_move(img).detected
