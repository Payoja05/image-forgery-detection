import json

import matplotlib.pyplot as plt
import streamlit as st

from config.settings import EVAL_EXAMPLES_DIR, HISTORY_PATH, METRICS_PATH
from ui.components import esc, note, page_header, section, show_table, sticky
from visualization import charts

TYPE_NAMES = {"splice_object": "Splicing (object)", "splice_region": "Splicing (region)",
              "copy_move": "Copy-move", "authentic": "Authentic"}


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def _num(v):
    return "—" if v is None else f"{v:.3f}"


def _plot(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


page_header("Model Performance", "Measured results of the trained model. Every number on this page is read from files "
            "written by the training and evaluation scripts - nothing is typed in by hand.", icon="chart")

metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else None

if metrics is None:
    note("<b>Test-set metrics are not available yet.</b> Train the model (<code>python -m training.train</code>) and "
         "then run <code>python -m evaluation.evaluate_model</code> to produce <code>reports/test_metrics.json</code>.", "warn")
else:
    il, pl = metrics["image_level"], metrics["pixel_level"]
    note(f"<b>Test set:</b> {esc(metrics['dataset'])} - {metrics['n_images']} images "
         f"({metrics['n_authentic']} authentic, {metrics['n_manipulated']} manipulated), never seen during training "
         f"or threshold calibration. Evaluated {esc(metrics['created'])}.", "blue")

    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    with c1, st.container(border=True):
        st.html('<div class="card-title">Image-level classification</div>')
        show_table([("Accuracy", _pct(il["accuracy"])), ("Precision", _pct(il["precision"])),
                    ("Recall", _pct(il["recall"])), ("F1 score", _pct(il["f1"])),
                    ("Specificity", _pct(il["specificity"])), ("ROC-AUC", _num(il.get("roc_auc")))])
        st.caption(f"Decision threshold {il['threshold']:.3f} on the image score ({esc(il.get('threshold_source', ''))}).")
    with c2, st.container(border=True):
        st.html('<div class="card-title">Confusion matrix</div>')
        _plot(charts.confusion_matrix_fig(il["confusion"]))
    with c3, st.container(border=True):
        st.html('<div class="card-title">ROC curve</div>')
        if "roc" in il:
            _plot(charts.roc_fig(il["roc"], il.get("roc_auc")))

    p1, p2 = st.columns([1, 1.2], gap="medium")
    mp = pl["mean_over_manipulated_images"]
    with p1, st.container(border=True):
        st.html('<div class="card-title">Localization performance (pixel level)</div>')
        show_table([("IoU (Intersection over Union)", _num(mp.get("iou"))), ("Dice coefficient", _num(mp.get("dice"))),
                    ("Pixel precision", _num(mp.get("precision"))), ("Pixel recall", _num(mp.get("recall"))),
                    ("Pixel ROC-AUC (sampled pixels)", _num(pl.get("pixel_roc_auc_sampled"))),
                    ("False-positive area on authentic images", _pct(pl.get("authentic_images_mean_false_positive_area")))])
        st.caption(f"Mean over the {pl['n_manipulated_images']} manipulated test images, pixel threshold {pl['threshold']}.")
    with p2, st.container(border=True):
        st.html('<div class="card-title">Per manipulation type</div>')
        rows = []
        for t, v in pl["per_type"].items():
            det_rate = metrics["image_level_per_type"].get(t, {}).get("detected_as_suspicious")
            rows.append((TYPE_NAMES.get(t, t), v["n"], _pct(det_rate), _num(v.get("iou")), _num(v.get("dice"))))
        auth = metrics["image_level_per_type"].get("authentic")
        if auth:
            rows.append(("Authentic", auth["n"], f"{_pct(auth['detected_as_suspicious'])} (false alarms)", "—", "—"))
        show_table(rows, ("Type", "Images", "Flagged suspicious", "Mean IoU", "Mean Dice"))
        t = metrics["timing"]
        st.caption(f"Mean inference time {t['mean_inference_seconds'] * 1000:.0f} ms per image on {t['device']} "
                   "(test images are 320×320).")

    if metrics.get("classical_baselines"):
        section("Classical forensics vs deep learning (same test images)")
        b = metrics["classical_baselines"]
        names = ["ELA", "Noise", "U-Net (ours)"]
        ious = [b["ela"].get("iou", 0), b["noise"].get("iou", 0), mp.get("iou", 0)]
        aucs = [b["ela"].get("pixel_roc_auc_sampled") or 0, b["noise"].get("pixel_roc_auc_sampled") or 0,
                pl.get("pixel_roc_auc_sampled") or 0]
        g1, g2 = st.columns([1.3, 1])
        with g1:
            _plot(charts.method_comparison_fig(names, ious, aucs))
        with g2:
            show_table([(b["ela"]["name"], _num(b["ela"].get("iou")), _num(b["ela"].get("dice"))),
                        (b["noise"]["name"], _num(b["noise"].get("iou")), _num(b["noise"].get("dice"))),
                        ("U-Net, threshold 0.5", _num(mp.get("iou")), _num(mp.get("dice")))],
                       ("Method", "Mean IoU", "Mean Dice"))
            st.caption("Classical maps are binarised per image with Otsu's threshold. This compares methods on the "
                       "synthetic test data only; it does not show which is better on real-world forgeries.")

    if metrics.get("examples"):
        section("Ground truth vs predicted masks")
        st.caption("The first two test images of each type in file order (not hand-picked).")
        for ex in metrics["examples"]:
            stem = ex["stem"]
            paths = [EVAL_EXAMPLES_DIR / f"{stem}_{k}.png" for k in ("image", "gt", "pred", "prob")]
            if not all(p.exists() for p in paths):
                continue
            cols = st.columns([1, 1, 1, 1, 0.9])
            for col, p, cap in zip(cols, paths, ("Image", "Ground truth", "Predicted mask", "Probability")):
                col.image(str(p), caption=cap, width="stretch")
            iou = "—" if ex["iou"] is None else f"{ex['iou']:.3f}"
            cols[4].html(f'<div class="caption-hand" style="text-align:left;margin-top:1.2rem">{TYPE_NAMES.get(ex["type"], ex["type"])}'
                         f'<br>score {ex["score"]:.3f}<br>IoU {iou}</div>')

# ------------------------------------------------------------- training history
section("Training history")
if history is None or not history.get("epochs"):
    note("No training history found (<code>reports/training_history.json</code>). It is written by "
         "<code>python -m training.train</code>.", "warn")
else:
    ep = history["epochs"]
    h1, h2, h3 = st.columns([1, 1, 0.7])
    with h1, st.container(border=True):
        st.html('<div class="card-title">Loss per epoch</div>')
        _plot(charts.loss_curve_fig(ep))
    with h2, st.container(border=True):
        st.html('<div class="card-title">Validation metrics per epoch</div>')
        _plot(charts.val_metrics_fig(ep))
    with h3:
        best = next(r for r in ep if r["epoch"] == history.get("best_epoch"))
        cfg = history["config"]
        show_table([("Epochs run", len(ep)), ("Best epoch (lowest val loss)", history.get("best_epoch")),
                    ("Best val loss", f"{best['val_loss']:.4f}"), ("Batch size", cfg.get("batch_size")),
                    ("Learning rate", cfg.get("lr")), ("Loss", cfg.get("loss")), ("Device", cfg.get("device")),
                    ("Time / epoch", f"{sum(r['seconds'] for r in ep) / len(ep) / 60:.1f} min")])
        sticky("validation ≠ test: test numbers are only in the tables above", "pink")
