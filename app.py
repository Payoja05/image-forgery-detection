"""Streamlit entry point:  streamlit run app.py

Uses Streamlit's native multipage API (st.navigation). The built-in nav is
hidden and replaced by a custom sidebar that matches the design reference.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui.components import icon_img, leaf  # noqa: E402
from ui.theme import apply_theme  # noqa: E402

st.set_page_config(page_title="Image Forgery Detection", page_icon="📷", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True, url_path="home"),
    st.Page("views/upload.py", title="Upload & Analyse", icon=":material/upload:", url_path="upload"),
    st.Page("views/forensics.py", title="Image Forensics", icon=":material/image_search:", url_path="forensics"),
    st.Page("views/results.py", title="Results & Visualizations", icon=":material/insights:", url_path="results"),
    st.Page("views/performance.py", title="Model Performance", icon=":material/bar_chart:", url_path="performance"),
    st.Page("views/about.py", title="About / Methodology", icon=":material/menu_book:", url_path="about"),
]
st.session_state["_pages"] = {p.url_path: p for p in PAGES}

current = st.navigation(PAGES, position="hidden")
apply_theme()

with st.sidebar:
    st.html(f'<div class="brand">{icon_img("camera", 34)}<div class="name">Image Forgery<br>Detection</div></div>')
    for page in PAGES:
        if page.url_path == current.url_path:
            symbol = page.icon.removeprefix(":material/").removesuffix(":")
            st.html(f'<div class="nav-active"><span class="material">{symbol}</span>{page.title}</div>')
        else:
            st.page_link(page, label=page.title, icon=page.icon)
    st.html(f'<div class="sidebar-leaf">{leaf()}</div>')

current.run()
