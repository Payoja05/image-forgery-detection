"""Image-level classification and pixel-level localisation metrics.

All functions are plain NumPy so they are easy to unit-test.

Pixel metrics for one image (pred, gt are boolean masks):
    TP = |pred & gt|, FP = |pred & ~gt|, FN = |~pred & gt|
    IoU       = TP / (TP + FP + FN)
    Dice (F1) = 2TP / (2TP + FP + FN)
    precision = TP / (TP + FP),  recall = TP / (TP + FN)
Convention for empty cases: if pred and gt are both empty the image is a
perfect match (all metrics = 1); if only one is empty the overlap metrics are 0.
"""
from __future__ import annotations

import numpy as np

# Note: ROC functions are implemented in NumPy instead of scikit-learn because
# scikit-learn's SciPy dependency is blocked by Windows Smart App Control on
# the development machine. They are unit-tested in tests/test_metrics.py.


def roc_curve(y_true, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False/true positive rates at every distinct score threshold (descending)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_group = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[last_of_group]
    fps = (last_of_group + 1) - tps
    tpr = np.r_[0.0, tps / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps / max(fps[-1], 1)]
    return fpr, tpr, np.r_[np.inf, s[last_of_group]]


def roc_auc_score(y_true, scores) -> float:
    """Area under the ROC curve (trapezoidal rule; ties handled correctly)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
        "fn": int((y_true & ~y_pred).sum()), "tn": int((~y_true & ~y_pred).sum()),
    }


def _safe(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(y_true, y_pred) -> dict:
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision, recall = _safe(tp, tp + fp), _safe(tp, tp + fn)
    return {
        "accuracy": _safe(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe(2 * precision * recall, precision + recall),
        "specificity": _safe(tn, tn + fp),
        "confusion": c,
    }


def roc_auc(y_true, scores) -> float | None:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def roc_points(y_true, scores, max_points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """Threshold on `scores` maximising F1 (used ONLY on the validation split)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(scores, 4)):
        f1 = classification_metrics(y_true, scores >= t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, float(best_f1)


def pixel_metrics(pred, gt) -> dict:
    pred, gt = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    if tp + fp + fn == 0:                       # both empty
        return {"iou": 1.0, "dice": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}
    dice = _safe(2 * tp, 2 * tp + fp + fn)
    return {
        "iou": _safe(tp, tp + fp + fn),
        "dice": dice,
        "f1": dice,
        "precision": _safe(tp, tp + fp) if tp + fp else 0.0,
        "recall": _safe(tp, tp + fn) if tp + fn else 0.0,
    }


def mean_pixel_metrics(per_image: list[dict]) -> dict:
    if not per_image:
        return {}
    keys = ("iou", "dice", "precision", "recall")
    return {k: float(np.mean([m[k] for m in per_image])) for k in keys}
