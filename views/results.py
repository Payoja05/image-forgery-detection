import cv2
import numpy as np
import streamlit as st

from config.settings import DEFAULT_PIXEL_THRESHOLD
from evaluation.evaluate_model import otsu_mask
from evaluation.metrics import pixel_metrics
from forensic_analysis.ela import ela_suspicion_map
from forensic_analysis.noise_analysis import noise_analysis
from ui.components import (
    esc, heatmap_legend, image_block, mask_legend, note, page_header, pill, section, show_table, sticky,
)
from ui.state import get_analysis, get_current, get_detector, run_analysis
from utils.report_export import build_report, png_bytes, report_json_bytes
from visualization.overlays import blend_heatmap, colorize, heatmap, overlay_mask

page_header("Analysis Results", "Model output for the selected image: image-level assessment, heatmap, mask and overlay.")

cur = get_current()
if cur is None:
    note("No image selected. Go to <b>Upload & Analyse</b> first.", "blue")
    if st.button("Go to Upload & Analyse", type="primary"):
        st.switch_page(st.session_state["_pages"]["upload"])
    st.stop()

analysis = get_analysis(cur)
if analysis is None:
    note(f"<b>{esc(cur.loaded.name)}</b> has not been analysed yet.", "blue")
    if st.button("Analyse now", type="primary"):
        with st.spinner("Running model inference ..."):
            analysis = run_analysis(cur)
        st.rerun()
    st.stop()
if analysis.get("error"):
    note(f"<b>The deep-learning model is unavailable, so no prediction was made.</b><br>{esc(analysis['error'])}", "warn")
    st.stop()

detector, _ = get_detector()
rgb = cur.loaded.rgb
prob = analysis["prob_map"]

# The slider is drawn next to the mask (below), but its value is needed here first.
thr_key = f"thr_{cur.key}"
thr = float(st.session_state.get(thr_key, DEFAULT_PIXEL_THRESHOLD))
det = detector.result_from_prob(prob, thr, analysis["inference_seconds"])
kind = "warn" if det.is_suspicious else "ok"

# ------------------------------------------------------------- summary
top_l, top_r = st.columns([3, 1])
with top_l, st.container(border=True):
    show_table([
        ("Assessment", pill(det.label, kind)),
        ("Image-level score", f"{det.image_score:.3f}  (decision threshold {det.image_threshold:.3f}, calibrated on validation data)"),
        ("Predicted tampered area", f"{100 * det.tampered_fraction:.2f}% of pixels at threshold {thr:.2f}"),
        ("Processing time", f"{det.inference_seconds:.2f} s model inference ({analysis['model_meta'].get('arch_name', '')}, CPU)"),
        ("Image", f"{esc(cur.loaded.name)} · {cur.loaded.width}×{cur.loaded.height}"),
    ])
with top_r:
    sticky("score = mean of the top 1% pixel probabilities", "")
    if cur.sample_label:
        sticky(f"ground truth: {cur.sample_label}", "pink" if cur.sample_label != "Authentic" else "sage")

annotated = overlay_mask(rgb, det.mask)
c1, c2, c3 = st.columns(3)
with c1:
    image_block(rgb, "Original image")
with c2:
    image_block(heatmap(prob), "Tampering localization", "(probability heatmap)")
    heatmap_legend()
with c3:
    image_block(annotated, "Detected tampered region", "(overlay at current threshold)")

m1, m2 = st.columns([1, 2], gap="large")
with m1:
    image_block(det.mask, "Segmentation mask", f"(threshold {thr:.2f})")
    mask_legend()
    st.slider("Segmentation threshold", 0.05, 0.95, DEFAULT_PIXEL_THRESHOLD, 0.05, key=thr_key,
              help="Pixels with probability ≥ threshold are marked as tampered. This re-thresholds the "
                   "model's real probability map; the model is not re-run.")
with m2:
    section("Interpretation")
    bullets = []
    if det.is_suspicious:
        bullets.append(f"The image-level score ({det.image_score:.3f}) is above the decision threshold "
                       f"({det.image_threshold:.3f}), so the model flags this image as <b>suspicious</b>.")
    else:
        bullets.append(f"The image-level score ({det.image_score:.3f}) is below the decision threshold "
                       f"({det.image_threshold:.3f}): the model found <b>no strong sign</b> of the kinds of "
                       "manipulation it was trained on. This is <i>not</i> proof that the image is authentic.")
    if det.tampered_fraction > 0:
        bullets.append(f"{100 * det.tampered_fraction:.2f}% of pixels exceed the threshold; they are outlined in red "
                       "in the overlay. Warm colours in the heatmap mark higher probability.")
    else:
        bullets.append("No region survives the current threshold - try lowering it to see weaker responses.")
    bullets.append("Localization can be inaccurate on images unlike the training data (e.g. screenshots, heavy "
                   "compression, professional retouching, or AI-generated content).")
    st.html("<ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>")
    note("A high score is not proof of manipulation and a low score is not proof of authenticity. "
         "Use the result to decide <i>where to look</i>, together with the classical forensic maps.", "info")

# ------------------------------------------------------------- ground truth (examples only)
if cur.ground_truth is not None:
    section("Ground truth vs prediction")
    st.caption("Available because this is a labelled example image from the held-out test split.")
    gt = cur.ground_truth
    pred = det.mask > 0
    cmp = rgb.copy() // 2 + 64
    cmp[gt & pred] = (94, 160, 90)        # true positive  - green
    cmp[~gt & pred] = (210, 80, 70)       # false positive - red
    cmp[gt & ~pred] = (80, 130, 200)      # missed         - blue
    g1, g2, g3, g4 = st.columns([1, 1, 1, 1.15])
    with g1:
        image_block((gt * 255).astype(np.uint8), "Ground-truth mask")
    with g2:
        image_block(det.mask, "Predicted mask")
    with g3:
        image_block(cmp, "Agreement", "green = hit · red = false alarm · blue = missed")
    with g4:
        if gt.any():
            pm = pixel_metrics(pred, gt)
            show_table([("IoU", pm["iou"]), ("Dice / F1", pm["dice"]), ("Pixel precision", pm["precision"]),
                        ("Pixel recall", pm["recall"])], ("Metric (this image)", "Value"))
        else:
            st.markdown(f"Authentic image: **{100 * pred.mean():.2f}%** of pixels falsely flagged.")

# ------------------------------------------------------------- classical vs deep learning
section("Classical forensics vs deep learning")
ela = ela_suspicion_map(rgb)
noise = cv2.GaussianBlur(noise_analysis(rgb).inconsistency, (15, 15), 0)
k1, k2, k3 = st.columns(3)
with k1:
    image_block(colorize(ela), "ELA (smoothed)", "JPEG re-compression differences")
with k2:
    image_block(colorize(noise), "Noise inconsistency", "local noise level vs. image median")
with k3:
    image_block(blend_heatmap(rgb, prob), "Deep-learning heatmap", "U-Net probability, blended")
if cur.ground_truth is not None and cur.ground_truth.any():
    gt = cur.ground_truth
    show_table([
        ("ELA (Otsu threshold)", pixel_metrics(otsu_mask(ela), gt)["iou"]),
        ("Noise inconsistency (Otsu threshold)", pixel_metrics(otsu_mask(noise), gt)["iou"]),
        (f"Deep-learning model (threshold {thr:.2f})", pixel_metrics(det.mask > 0, gt)["iou"]),
    ], ("Method", "IoU with ground truth (this image only)"))
    st.caption("A single image says little; see Model Performance for the comparison over the whole test split.")
note("Classical maps respond to <i>any</i> local difference in compression or noise (including edges and texture), "
     "while the network learned which patterns tend to accompany pasted regions in its training data. They often "
     "disagree; neither should be trusted alone.", "blue")

# ------------------------------------------------------------- exports
section("Download results")
forensic_summary = {"ela_quality": 90, "ela_mean_suspicion": round(float(ela.mean()), 4),
                    "noise_mean_inconsistency": round(float(noise.mean()), 4)}
report = build_report(
    cur.loaded.name,
    {"width": cur.loaded.width, "height": cur.loaded.height, "format": cur.loaded.format,
     "size_bytes": cur.loaded.size_bytes, "source": cur.source, "ground_truth_label": cur.sample_label},
    det, {"arch": analysis["model_meta"].get("arch_name"), "epoch": analysis["model_meta"].get("epoch"),
          "trained_on": analysis["model_meta"].get("trained_on")},
    forensics=forensic_summary,
)
report["analysed_at"] = analysis["timestamp"]
stem = cur.loaded.name.rsplit(".", 1)[0]
d1, d2, d3, d4 = st.columns(4)
d1.download_button("Annotated image (PNG)", png_bytes(annotated), f"{stem}_annotated.png", "image/png", width="stretch")
d2.download_button("Segmentation mask (PNG)", png_bytes(det.mask), f"{stem}_mask.png", "image/png", width="stretch")
d3.download_button("Heatmap (PNG)", png_bytes(heatmap(prob)), f"{stem}_heatmap.png", "image/png", width="stretch")
d4.download_button("Report (JSON)", report_json_bytes(report), f"{stem}_report.json", "application/json", width="stretch")
