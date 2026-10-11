"""Evaluate the trained model (and classical baselines) on a held-out split.

    python -m evaluation.evaluate_model                      # synthetic test split
    python -m evaluation.evaluate_model --split-dir data/casia2_prepared --name "CASIA v2"

The split directory must follow the layout in training/dataset.py. An image is
treated as manipulated (label 1) if its ground-truth mask has any tampered pixel.

What is measured
  image level : accuracy / precision / recall / F1 / confusion matrix at the
                threshold calibrated on the VALIDATION split (stored in the
                checkpoint), plus threshold-free ROC-AUC.
  pixel level : IoU, Dice, pixel precision, pixel recall at pixel threshold 0.5,
                averaged over manipulated images; pixel ROC-AUC on sampled pixels;
                false-positive area on authentic images.
  baselines   : the same pixel metrics for ELA and noise-inconsistency maps
                (binarised per image with Otsu's threshold).
  timing      : mean model inference time per image on this machine.
Results are written to reports/test_metrics.json and shown in the app.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.settings import (  # noqa: E402
    CHECKPOINT_PATH, DEFAULT_PIXEL_THRESHOLD, EVAL_EXAMPLES_DIR, IMAGE_SCORE_TOP_FRACTION, METRICS_PATH, SEED, SYNTH_DIR,
)
from evaluation.metrics import (  # noqa: E402
    classification_metrics, mean_pixel_metrics, pixel_metrics, roc_auc, roc_points,
)
from forensic_analysis.ela import ela_suspicion_map  # noqa: E402
from forensic_analysis.noise_analysis import noise_analysis  # noqa: E402
from inference.localization import image_level_score, mask_from_probability, predict_probability_map  # noqa: E402
from models.model_loader import load_model  # noqa: E402
from training.dataset import load_pair, read_records  # noqa: E402

PIXELS_PER_IMAGE = 2000


def otsu_mask(m: np.ndarray) -> np.ndarray:
    u8 = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    _, b = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=SYNTH_DIR / "test")
    ap.add_argument("--name", default="Synthetic tampering test split (Oxford-IIIT Pet sources)")
    ap.add_argument("--out", type=Path, default=METRICS_PATH)
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--examples-per-type", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    model, meta = load_model(CHECKPOINT_PATH)
    img_thr = float(meta.get("image_threshold", 0.5))
    records = read_records(args.split_dir)
    print(f"Evaluating {len(records)} images from {args.split_dir}")

    labels, scores, times, types = [], [], [], []
    per_img, per_type = [], defaultdict(list)
    auth_fp_area = []
    px = {"gt": [], "model": [], "ela": [], "noise": []}
    classical = {"ela": [], "noise": []}
    examples, ex_count = [], defaultdict(int)
    EVAL_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for i, rec in enumerate(records):
        rgb, gt = load_pair(args.split_dir, rec)
        gt = gt > 0.5
        t0 = time.perf_counter()
        prob = predict_probability_map(model, rgb)
        times.append(time.perf_counter() - t0)
        score = image_level_score(prob, IMAGE_SCORE_TOP_FRACTION)
        pred = mask_from_probability(prob, DEFAULT_PIXEL_THRESHOLD) > 0
        label = int(gt.any())
        kind = rec.get("type") or ("manipulated" if label else "authentic")
        labels.append(label)
        scores.append(score)
        types.append(kind)

        idx = rng.choice(gt.size, size=min(PIXELS_PER_IMAGE, gt.size), replace=False)
        px["gt"].append(gt.ravel()[idx])
        px["model"].append(prob.ravel()[idx])

        if label:
            m = pixel_metrics(pred, gt)
            per_img.append(m)
            per_type[kind].append(m)
        else:
            auth_fp_area.append(float(pred.mean()))

        if not args.no_classical:
            ela = ela_suspicion_map(rgb)
            noise = cv2.GaussianBlur(noise_analysis(rgb).inconsistency, (15, 15), 0)
            px["ela"].append(ela.ravel()[idx])
            px["noise"].append(noise.ravel()[idx])
            if label:
                classical["ela"].append(pixel_metrics(otsu_mask(ela), gt))
                classical["noise"].append(pixel_metrics(otsu_mask(noise), gt))

        # first N images of each type in file order (not cherry-picked)
        if ex_count[kind] < args.examples_per_type:
            ex_count[kind] += 1
            stem = f"{kind}_{ex_count[kind]}"
            cv2.imwrite(str(EVAL_EXAMPLES_DIR / f"{stem}_image.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(EVAL_EXAMPLES_DIR / f"{stem}_gt.png"), gt.astype(np.uint8) * 255)
            cv2.imwrite(str(EVAL_EXAMPLES_DIR / f"{stem}_pred.png"), pred.astype(np.uint8) * 255)
            cv2.imwrite(str(EVAL_EXAMPLES_DIR / f"{stem}_prob.png"), (prob * 255).astype(np.uint8))
            examples.append({"stem": stem, "id": rec["id"], "type": kind, "score": round(score, 4),
                             "iou": round(pixel_metrics(pred, gt)["iou"], 4) if label else None})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)}")

    labels_a, scores_a = np.array(labels), np.array(scores)
    image_level = classification_metrics(labels_a, scores_a >= img_thr)
    image_level.update({"threshold": img_thr, "threshold_source": meta.get("image_threshold_source", ""),
                        "roc_auc": roc_auc(labels_a, scores_a)})
    if len(np.unique(labels_a)) == 2:
        image_level["roc"] = roc_points(labels_a, scores_a)
    per_type_image = {}
    for t in sorted(set(types)):
        sel = np.array(types) == t
        per_type_image[t] = {"n": int(sel.sum()), "detected_as_suspicious": float((scores_a[sel] >= img_thr).mean())}

    gt_px = np.concatenate(px["gt"])
    pixel_level = {
        "threshold": DEFAULT_PIXEL_THRESHOLD,
        "n_manipulated_images": len(per_img),
        "mean_over_manipulated_images": mean_pixel_metrics(per_img),
        "pixel_roc_auc_sampled": roc_auc(gt_px, np.concatenate(px["model"])),
        "authentic_images_mean_false_positive_area": float(np.mean(auth_fp_area)) if auth_fp_area else None,
        "per_type": {t: {"n": len(v), **mean_pixel_metrics(v)} for t, v in sorted(per_type.items())},
    }
    baselines = {}
    if not args.no_classical:
        for k, label_name in (("ela", "Error Level Analysis (Otsu)"), ("noise", "Noise inconsistency (Otsu)")):
            baselines[k] = {"name": label_name, **mean_pixel_metrics(classical[k]),
                            "pixel_roc_auc_sampled": roc_auc(gt_px, np.concatenate(px[k]))}

    out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.name,
        "split_dir": str(args.split_dir.relative_to(ROOT) if args.split_dir.is_relative_to(ROOT) else args.split_dir),
        "n_images": len(records),
        "n_authentic": int((labels_a == 0).sum()),
        "n_manipulated": int((labels_a == 1).sum()),
        "model": {"arch": meta.get("arch_name"), "epoch": meta.get("epoch"), "trained_on": meta.get("trained_on")},
        "image_level": image_level,
        "image_level_per_type": per_type_image,
        "pixel_level": pixel_level,
        "classical_baselines": baselines,
        "timing": {"mean_inference_seconds": float(np.mean(times)), "median_inference_seconds": float(np.median(times)),
                   "device": str(next(model.parameters()).device)},
        "examples": examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("image_level", "pixel_level", "classical_baselines", "timing")}, indent=2, default=str))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
