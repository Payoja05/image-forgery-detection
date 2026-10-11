import streamlit as st

from forensic_analysis.copy_move import detect_copy_move
from forensic_analysis.edge_analysis import edge_analysis
from forensic_analysis.ela import error_level_analysis
from forensic_analysis.noise_analysis import noise_analysis
from ui.components import esc, image_block, note, page_header, section, sticky
from ui.state import get_current
from visualization.overlays import colorize, gray_to_rgb


@st.cache_data(show_spinner=False, max_entries=8)
def _ela(key, _rgb, quality):
    return error_level_analysis(_rgb, quality)


@st.cache_data(show_spinner=False, max_entries=8)
def _noise(key, _rgb, block):
    return noise_analysis(_rgb, block)


@st.cache_data(show_spinner=False, max_entries=8)
def _edges(key, _rgb, low, high):
    return edge_analysis(_rgb, low, high)


@st.cache_data(show_spinner=False, max_entries=8)
def _copy_move(key, _rgb, ratio, min_dist, min_inliers):
    return detect_copy_move(_rgb, ratio, min_dist, min_inliers)


page_header("Image Forensics Analysis", icon="magnifier")
note("These are <b>traditional image-processing techniques</b> that expose possible signs of manipulation. They are "
     "shown alongside the deep-learning model, not instead of it, and every map needs human interpretation.", "info")

cur = get_current()
if cur is None:
    note("No image selected. Choose or upload one on <b>Upload & Analyse</b>.", "blue")
    if st.button("Go to Upload & Analyse", type="primary"):
        st.switch_page(st.session_state["_pages"]["upload"])
    st.stop()

rgb, key = cur.loaded.rgb, cur.key
st.caption(f"Analysing **{cur.loaded.name}** · {cur.loaded.width}×{cur.loaded.height} · {cur.loaded.format}")

# ------------------------------------------------------------------ ELA
section("A. Error Level Analysis (ELA)")
c1, c2, c3 = st.columns([1, 1, 0.55], gap="medium")
with c3:
    quality = st.slider("Re-save JPEG quality", 50, 99, 90, key="ela_q")
    st.caption("The image is re-compressed at this quality and subtracted from the original. "
               "The difference is brightness-stretched for display.")
ela = _ela(key, rgb, quality)
with c1:
    image_block(rgb, "Original image")
with c2:
    image_block(ela.visual, "ELA difference", f"max diff {ela.max_diff} · mean {ela.mean_diff:.2f}")
with c3:
    sticky("bright = changed more when re-saved", "pink")
note("ELA highlights regions with a different compression history. Edges, fine texture and saturated colours are "
     "also bright, and a PNG or a re-saved JPEG may look uniform - so ELA is <b>not a reliable detector on its own</b>.", "warn")

# ------------------------------------------------------------------ noise + copy-move
n_col, cm_col = st.columns(2, gap="large")
with n_col:
    section("B. Noise analysis")
    block = st.select_slider("Block size (px)", [8, 16, 24, 32], 16, key="noise_block")
    nz = _noise(key, rgb, block)
    t1, t2 = st.columns(2)
    with t1:
        image_block(gray_to_rgb(nz.residual_visual), "Noise residual", "image − median filter")
    with t2:
        image_block(colorize(nz.inconsistency), "Noise inconsistency", "robust z-score of local level")
    st.caption(f"Median local noise level: {nz.global_noise:.2f} (grey levels).")
    note("Pasted regions can carry a different noise level. But flat areas (sky) are naturally quiet and textured "
         "areas noisy, so the map shows <i>content</i> as well as possible tampering.", "blue")

with cm_col:
    section("C. Copy-move analysis")
    a, b, c = st.columns(3)
    ratio = a.slider("g2NN ratio", 0.3, 0.8, 0.5, 0.05, key="cm_ratio")
    min_dist = b.slider("Min distance (px)", 10, 100, 30, 5, key="cm_dist")
    min_inl = c.slider("Min inliers", 4, 30, 8, key="cm_inl")
    with st.spinner("Matching SIFT keypoints ..."):
        cm = _copy_move(key, rgb, ratio, float(min_dist), min_inl)
    image_block(cm.visual, "Keypoint matches", "blue = source · red = copy · yellow = match")
    if cm.detected:
        note(f"<b>Possible duplicated region:</b> {cm.num_inliers} geometrically consistent matches "
             f"(from {cm.num_candidate_matches} candidates, {cm.num_keypoints} keypoints).", "warn")
    else:
        note(f"No consistent duplicated region found ({cm.num_inliers} inliers, {cm.num_candidate_matches} candidate "
             f"matches, {cm.num_keypoints} keypoints).", "ok")
    st.caption("Fails on small/flat cloned areas and heavily transformed copies; repetitive scenes (windows, tiles) "
               "can produce genuine look-alike matches. Only the largest consistent cluster is reported.")

# ------------------------------------------------------------------ edges
section("D. Edge & sharpness analysis")
e1, e2, e3 = st.columns([1, 1, 0.55], gap="medium")
with e3:
    low = st.slider("Canny low threshold", 10, 200, 60, key="canny_lo")
    high = st.slider("Canny high threshold", 50, 400, 160, key="canny_hi")
    if high <= low:
        st.warning("High threshold should be greater than low.")
ed = _edges(key, rgb, low, max(high, low + 1))
with e1:
    image_block(gray_to_rgb(ed.edges), "Canny edges", f"edge density {100 * ed.edge_density:.1f}%")
with e2:
    image_block(colorize(ed.sharpness), "Local sharpness", "block-wise Laplacian variance")
with e3:
    sticky("look for objects sharper or softer than their surroundings", "blue")
note("A pasted object may have unnaturally crisp or blurred outlines. Depth of field and motion blur produce the same "
     "effect in genuine photos, so edge maps support - never replace - other evidence.", "blue")
