import csv
import io

import streamlit as st

from config.settings import MAX_FILE_SIZE_MB, SAMPLES_DIR
from preprocessing.image_preprocessing import image_quality_report
from ui.components import esc, note, page_header, section, show_table, sticky
from ui.state import get_current, get_detector, list_samples, load_sample, run_analysis, set_current
from utils.image_validation import ImageValidationError, validate_and_load

page_header("Upload an Image", "Upload a JPG or PNG to analyse it, or pick one of the labelled example images.", icon="upload")

tab_single, tab_batch = st.tabs(["Single image", "Batch analysis"])

# ------------------------------------------------------------------ single
with tab_single:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        up = st.file_uploader(f"Drag and drop an image here (JPG, JPEG, PNG · max {MAX_FILE_SIZE_MB} MB)",
                              type=["jpg", "jpeg", "png"], key="single_upload")
        if up is None:
            st.session_state.pop("_upload_error", None)
        elif st.session_state.get("_last_upload") != up.file_id:
            # only act on a *new* upload, so choosing an example afterwards is not overridden
            st.session_state["_last_upload"] = up.file_id
            try:
                set_current(validate_and_load(up.getvalue(), up.name), "upload")
                st.session_state.pop("_upload_error", None)
            except ImageValidationError as exc:
                st.session_state["_upload_error"] = str(exc)
        if st.session_state.get("_upload_error"):
            note(f"<b>Could not use this file.</b> {esc(st.session_state['_upload_error'])}", "warn")

        samples = list_samples()
        section("Example images")
        if samples:
            st.caption("Held-out TEST images from the synthetic dataset (Oxford-IIIT Pet sources, CC BY-SA 4.0). "
                       "Labels and masks are ground truth, not model output.")
            cols = st.columns(len(samples))
            for col, s in zip(cols, samples):
                with col:
                    st.image(str(SAMPLES_DIR / s["file"]), width="stretch")
                    st.html(f'<div class="caption-hand">{esc(s["label"])}<br><small>{esc(s["type_label"])}</small></div>')
                    if st.button("Use", key=f"use_{s['file']}", width="stretch"):
                        load_sample(s)
                        st.rerun()
        else:
            note("No example images found in <code>assets/samples</code>. Run "
                 "<code>python -m training.make_samples</code> after building the dataset.", "blue")

    with right:
        cur = get_current()
        if cur is None:
            note("No image selected yet. Upload a file or choose an example on the left.", "blue")
        else:
            img = cur.loaded
            with st.container(border=True):
                st.image(img.rgb, caption=img.name, width="stretch")
                q = image_quality_report(img.rgb)
                rows = [("Dimensions", f"{img.width} × {img.height} px"), ("File size", f"{img.size_kb:.1f} KB"),
                        ("Format", img.format), ("Source", "example (ground truth available)" if cur.source == "sample" else "your upload")]
                if cur.sample_label:
                    rows.append(("Ground-truth label", cur.sample_label))
                show_table(rows)
                warn = [m for flag, m in ((q["is_low_resolution"], "low resolution"), (q["is_very_blurry"], "very blurry"),
                                          (q["is_near_uniform"], "almost uniform")) if flag]
                if warn:
                    note("Image quality: " + ", ".join(warn) + ". Results may be less reliable.", "warn")
                if img.format == "PNG":
                    st.caption("PNG is lossless: Error Level Analysis is less informative unless the image was a JPEG before.")

            if st.button("Analyse image", type="primary", width="stretch"):
                with st.spinner("Running preprocessing and model inference ..."):
                    res = run_analysis(cur)
                if res.get("error"):
                    note(f"<b>Deep-learning model unavailable.</b> {esc(res['error'])}<br>"
                         "The classical forensic tools still work on the <b>Image Forensics</b> page.", "warn")
                else:
                    st.switch_page(st.session_state["_pages"]["results"])
            sticky("the model runs locally on your CPU — nothing is uploaded anywhere", "sage")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.markdown("Analyse several images at once with the deep-learning model. Each file is validated separately; "
                "a failing file does not stop the batch.")
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                             key="batch_upload")
    if files and st.button(f"Analyse {len(files)} image(s)", type="primary"):
        detector, err = get_detector()
        if detector is None:
            note(f"<b>Model unavailable:</b> {esc(err)}", "warn")
        else:
            rows, bar = [], st.progress(0.0)
            for i, f in enumerate(files, 1):
                row = {"file": f.name, "width": "", "height": "", "assessment": "", "image_score": "",
                       "predicted_tampered_area_%": "", "inference_s": "", "error": ""}
                try:
                    img = validate_and_load(f.getvalue(), f.name)
                    det = detector.detect(img.rgb)
                    row.update(width=img.width, height=img.height, assessment=det.label,
                               image_score=round(det.image_score, 4),
                               **{"predicted_tampered_area_%": round(100 * det.tampered_fraction, 2)},
                               inference_s=round(det.inference_seconds, 3))
                except ImageValidationError as exc:
                    row["error"] = str(exc)
                except Exception as exc:          # keep going for the remaining files
                    row["error"] = f"analysis failed: {exc}"
                rows.append(row)
                bar.progress(i / len(files))
            st.session_state["batch_rows"] = rows
            st.session_state["batch_threshold"] = detector.image_threshold
    rows = st.session_state.get("batch_rows")
    if rows:
        st.caption(f"Image-level decision threshold: {st.session_state['batch_threshold']:.3f} "
                   "(calibrated on the validation split). Pixel threshold 0.5.")
        st.dataframe(rows, width="stretch", hide_index=True)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        st.download_button("Download results (CSV)", buf.getvalue(), "batch_results.csv", "text/csv")
