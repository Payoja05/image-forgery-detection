import numpy as np
import pytest

from evaluation.metrics import (
    best_f1_threshold, classification_metrics, confusion, pixel_metrics, roc_auc, roc_auc_score, roc_curve,
)


def test_confusion_and_classification():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    p = [1, 0, 1, 0, 1, 0, 0, 1]
    c = confusion(y, p)
    assert c == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    m = classification_metrics(y, p)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)


def test_classification_no_positives_predicted():
    m = classification_metrics([1, 0], [0, 0])
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0


def test_roc_auc_known_values():
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc_score([0, 1], [0.2, 0.9]) == pytest.approx(1.0)
    assert roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)   # ties
    assert roc_auc([1, 1], [0.2, 0.3]) is None


def test_roc_auc_matches_rank_statistic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    pos, neg = s[y == 1], s[y == 0]
    mann_whitney = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert roc_auc_score(y, s) == pytest.approx(mann_whitney)
    fpr, tpr, _ = roc_curve(y, s)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1


def test_best_f1_threshold_separable():
    t, f1 = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
    assert f1 == 1.0 and 0.2 < t <= 0.7


def test_pixel_metrics():
    gt = np.zeros((10, 10), bool)
    gt[:5] = True                         # 50 px
    pred = np.zeros((10, 10), bool)
    pred[2:7] = True                      # 50 px, 30 overlap
    m = pixel_metrics(pred, gt)
    assert m["iou"] == pytest.approx(30 / 70)
    assert m["dice"] == pytest.approx(60 / 100)
    assert m["precision"] == pytest.approx(0.6) and m["recall"] == pytest.approx(0.6)


def test_pixel_metrics_empty_cases():
    z = np.zeros((4, 4), bool)
    o = np.ones((4, 4), bool)
    assert pixel_metrics(z, z)["iou"] == 1.0
    assert pixel_metrics(o, z)["iou"] == 0.0 and pixel_metrics(z, o)["recall"] == 0.0
