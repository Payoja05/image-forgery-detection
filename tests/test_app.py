"""Headless integration tests of the Streamlit UI (streamlit.testing.AppTest).

They render every page and drive the real flow: choose an example image ->
analyse with the real model -> results page. Model-dependent tests are skipped
if no trained checkpoint exists.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config.settings import CHECKPOINT_PATH, SAMPLES_DIR

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")
PAGES = ["views/home.py", "views/upload.py", "views/forensics.py", "views/results.py",
         "views/performance.py", "views/about.py"]


def _app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page):
    at = _app()
    at.switch_page(page)
    at.run()
    assert not at.exception, at.exception


def _use_first_sample(at, kind_index=2):
    at.switch_page("views/upload.py")
    at.run()
    use_buttons = [b for b in at.button if b.label == "Use"]
    assert use_buttons, "example images missing"
    use_buttons[kind_index].click()
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.mark.skipif(not (SAMPLES_DIR / "samples.json").exists(), reason="no example images")
def test_forensics_page_with_example_image():
    at = _use_first_sample(_app(), kind_index=1)          # copy-move example
    at.switch_page("views/forensics.py")
    at.run()
    assert not at.exception, at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "Analysing **test_" in captions                     # the selected example is analysed
    assert "largest consistent cluster" in captions            # copy-move section rendered


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="model not trained yet")
def test_analyse_and_results_flow():
    at = _use_first_sample(_app(), kind_index=2)          # object-splicing example
    next(b for b in at.button if b.label == "Analyse image").click()
    at.run()
    assert not at.exception, at.exception
    analysis = at.session_state["analysis"]
    assert analysis["error"] is None and analysis["prob_map"].shape == (320, 320)
    at.switch_page("views/results.py")
    at.run()
    assert not at.exception, at.exception
    slider = at.slider(key=f"thr_{at.session_state['current'].key}")
    slider.set_value(0.8).run()                         # interactive threshold re-renders without error
    assert not at.exception, at.exception
