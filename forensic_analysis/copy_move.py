"""Keypoint-based copy-move (cloning) detection.

Pipeline (after Amerini et al., "A SIFT-based forensic method for copy-move
attack detection", IEEE TIFS 2011):
 1. Detect SIFT keypoints/descriptors on the grayscale image.
 2. Match every descriptor against all others in the *same* image.
 3. Generalised 2NN test (g2NN): keep neighbours while d_i / d_{i+1} < ratio,
    skipping the trivial self-match.
 4. Discard pairs that are spatially too close (same structure, not a copy).
 5. Fit a geometric transform with RANSAC; a cluster with enough inliers is
    reported as a suspected duplicated region pair.

Failure cases: very small or flat cloned regions (few keypoints), heavy
resizing/blurring of the clone, and naturally repetitive scenes (windows,
tiles, crowds) which create genuine duplicate-looking matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CopyMoveResult:
    visual: np.ndarray                      # RGB image with matches drawn
    mask: np.ndarray                        # H x W uint8 (0/255) hull regions
    num_keypoints: int
    num_candidate_matches: int
    num_inliers: int
    detected: bool
    pairs: list = field(default_factory=list)   # [(x1,y1,x2,y2), ...] inlier pairs


def _sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02)
    raise RuntimeError("This OpenCV build has no SIFT (need opencv >= 4.4).")


def detect_copy_move(
    rgb: np.ndarray,
    ratio: float = 0.5,
    min_distance: float = 30.0,
    min_inliers: int = 8,
    max_side: int = 1024,
) -> CopyMoveResult:
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    work = cv2.resize(rgb, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    kps, desc = _sift().detectAndCompute(gray, None)
    visual = rgb.copy()
    empty_mask = np.zeros((h0, w0), np.uint8)
    if desc is None or len(kps) < 3:
        return CopyMoveResult(visual, empty_mask, len(kps or []), 0, 0, False)

    pts = np.array([kp.pt for kp in kps], np.float32)
    k = min(10, len(kps))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc, desc, k=k)

    src, dst = [], []
    seen = set()
    for i, neigh in enumerate(knn):
        neigh = [m for m in neigh if m.trainIdx != i]           # drop self-match
        for a, b in zip(neigh, neigh[1:]):                       # g2NN
            if a.distance >= ratio * max(b.distance, 1e-6):
                break
            j = a.trainIdx
            if np.linalg.norm(pts[i] - pts[j]) < min_distance * scale:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            src.append(pts[i])
            dst.append(pts[j])

    n_cand = len(src)
    inl_src, inl_dst = np.empty((0, 2)), np.empty((0, 2))
    if n_cand >= min_inliers:
        src_a, dst_a = np.array(src), np.array(dst)
        # Pairs may be stored in either direction; orient them consistently
        # (left-most point first) so one transform can explain the cluster.
        swap = src_a[:, 0] > dst_a[:, 0]
        src_a[swap], dst_a[swap] = dst_a[swap].copy(), src_a[swap].copy()
        M, inliers = cv2.estimateAffinePartial2D(src_a, dst_a, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is not None and inliers is not None:
            sel = inliers.ravel().astype(bool)
            inl_src, inl_dst = src_a[sel] / scale, dst_a[sel] / scale

    detected = len(inl_src) >= min_inliers
    mask = empty_mask.copy()
    if detected:
        for group, colour in ((inl_src, (66, 133, 200)), (inl_dst, (200, 70, 60))):
            hull = cv2.convexHull(group.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            cv2.polylines(visual, [hull], True, colour, max(2, w0 // 300))
    lw = max(1, w0 // 500)
    for (x1, y1), (x2, y2) in zip(inl_src, inl_dst):
        cv2.line(visual, (int(x1), int(y1)), (int(x2), int(y2)), (230, 180, 40), lw)
        cv2.circle(visual, (int(x1), int(y1)), lw + 2, (66, 133, 200), -1)
        cv2.circle(visual, (int(x2), int(y2)), lw + 2, (200, 70, 60), -1)

    pairs = [(float(a[0]), float(a[1]), float(b[0]), float(b[1])) for a, b in zip(inl_src, inl_dst)]
    return CopyMoveResult(visual, mask, len(kps), n_cand, len(inl_src), detected, pairs)
