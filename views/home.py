import streamlit as st

from ui.components import feature_cards, leaf, note, page_header, pipeline, section, sticky

left, right = st.columns([5, 1.1])
with left:
    page_header(
        "Image Forgery Detection and Tampering Localization",
        "A computer vision project for investigating digital image manipulation. Upload an image to check "
        "it for signs of manipulation and see <i>where</i> the suspicious regions are.",
        icon="camera",
    )
with right:
    sticky("traditional forensics + deep learning, side by side", "blue")

feature_cards([
    ("layers", "Image preprocessing", "Validation, colour conversion, resizing and normalisation that keep forensic traces intact.", "sage"),
    ("magnifier", "Forgery detection", "An image-level score that flags pictures showing signs of splicing or cloning.", "blue"),
    ("dashed_box", "Tampering localization", "A U-Net predicts a per-pixel probability map and a binary mask of suspicious regions.", "butter"),
    ("document", "Forensic visualization", "Error Level Analysis, noise, edge and copy-move maps for visual inspection.", "pink"),
])

c1, c2 = st.columns([1.35, 1], gap="large")
with c1:
    section("The problem")
    st.markdown(
        "Image editing tools make it easy to **splice** an object from one photo into another, "
        "**copy-move** a region within the same photo, or **remove** something entirely. Such edits are often "
        "invisible to the eye, yet they leave statistical traces — different JPEG compression history, "
        "inconsistent sensor noise, unnatural boundaries or duplicated texture."
    )
    section("Objectives")
    st.markdown(
        "1. Detect whether an image shows signs of manipulation (image-level).\n"
        "2. Localize the suspicious region at pixel level and produce a segmentation mask.\n"
        "3. Visualise the evidence with heatmaps and overlays.\n"
        "4. Compare with classical forensic techniques (ELA, noise, edges, copy-move).\n"
        "5. Evaluate honestly on held-out data with standard metrics."
    )
    section("Why does this matter?")
    st.markdown(
        "Manipulated images are used in misinformation, insurance and document fraud, fake evidence and "
        "scientific misconduct. Automated tools cannot replace a human expert, but they can point an analyst "
        "to the regions that deserve a closer look."
    )
with c2:
    section("How it works")
    pipeline([
        ("Input image", "JPG / PNG upload", "pink"),
        ("Validation & preprocessing", "format, size, resize, normalise", "blue"),
        ("Classical forensics", "ELA · noise · edges · copy-move", "butter"),
        ("Deep learning model", "U-Net + SRM noise stream", "sage"),
        ("Localization & results", "heatmap · mask · report", "blue"),
    ])
    st.write("")
    if st.button("Start analysing an image  →", type="primary", width="stretch"):
        st.switch_page(st.session_state["_pages"]["upload"])

note(
    "<b>Limitations.</b> The model was trained on a <i>synthetic</i> forgery dataset (see Methodology), so it "
    "works best on splicing / copy-move edits similar to that data and may miss subtle or professional edits. "
    "A high score is not proof of manipulation, and a low score is not proof of authenticity.",
    "warn",
)
st.html(f'<div style="display:flex;justify-content:flex-end;opacity:.7;margin-top:-1rem">{leaf()}</div>')
