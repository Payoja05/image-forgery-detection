"""Reusable UI building blocks (headings, cards, notes, tables, pipeline)."""
from __future__ import annotations

import base64
import html

import numpy as np
import streamlit as st

# ---------------------------------------------------------------- line icons
_S = 'fill="none" stroke="#292929" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'
ICONS = {
    "camera": f'<svg viewBox="0 0 48 48" {_S}><path d="M8 16h8l3-5h10l3 5h8v22H8z"/><circle cx="24" cy="26" r="7"/><circle cx="24" cy="26" r="3.2"/><path d="M35 20h2"/></svg>',
    "magnifier": f'<svg viewBox="0 0 48 48" {_S}><circle cx="20" cy="20" r="11"/><path d="M28 28l11 11"/><path d="M14.5 16.5c1.4-2.4 3.6-3.6 6-3.8"/><path d="M13 22l4-3 3 4 4-6 3 4"/></svg>',
    "dashed_box": f'<svg viewBox="0 0 48 48" {_S}><path d="M10 10h6M22 10h6M34 10h4v4M38 20v6M38 32v6h-4M28 38h-6M16 38h-6v-4M10 28v-6M10 16v-6"/></svg>',
    "chart": f'<svg viewBox="0 0 48 48" {_S}><path d="M8 40h32"/><rect x="12" y="26" width="5" height="12"/><rect x="21" y="18" width="5" height="20"/><rect x="30" y="10" width="5" height="28"/></svg>',
    "document": f'<svg viewBox="0 0 48 48" {_S}><path d="M13 7h15l8 8v26H13z"/><path d="M28 7v8h8"/><path d="M18 22h13M18 27h13M18 32h9"/></svg>',
    "layers": f'<svg viewBox="0 0 48 48" {_S}><path d="M24 9l16 8-16 8-16-8z"/><path d="M8 25l16 8 16-8"/><path d="M8 32l16 8 16-8"/></svg>',
    "gear": f'<svg viewBox="0 0 48 48" {_S}><circle cx="24" cy="24" r="6"/><path d="M24 8v5M24 35v5M8 24h5M35 24h5M12.7 12.7l3.5 3.5M31.8 31.8l3.5 3.5M12.7 35.3l3.5-3.5M31.8 16.2l3.5-3.5"/></svg>',
    "upload": f'<svg viewBox="0 0 48 48" {_S}><path d="M15 33h-2a8 8 0 0 1-1-15.9A11 11 0 0 1 33 15a8 8 0 0 1 2 18h-2"/><path d="M24 22v16M18 28l6-6 6 6"/></svg>',
}

LEAF = (
    '<svg width="90" height="120" viewBox="0 0 90 120" fill="none" stroke="#7C9A6E" stroke-width="1.5" stroke-linecap="round">'
    '<path d="M20 118 C30 90 45 60 70 12"/>'
    '<path d="M30 92c-14-2-20-12-20-20 10 0 19 8 20 20z" fill="#E4EDDF"/>'
    '<path d="M37 76c13-4 19-13 19-21-10 1-18 9-19 21z" fill="#E4EDDF"/>'
    '<path d="M44 62c-13-3-18-12-18-20 10 1 17 9 18 20z" fill="#E4EDDF"/>'
    '<path d="M52 46c12-5 17-14 16-22-9 2-16 10-16 22z" fill="#E4EDDF"/>'
    '<path d="M58 34c-11-4-15-12-14-19 9 2 14 9 14 19z" fill="#E4EDDF"/></svg>'
)

SQUIGGLE = (
    '<svg class="squiggle" width="{w}" height="10" viewBox="0 0 {w} 10" fill="none">'
    '<path d="M2 6 {path}" stroke="#7FA6C9" stroke-width="2.2" stroke-linecap="round"/></svg>'
)


def _squiggle(width: int = 170) -> str:
    segs, x = [], 2
    while x < width - 12:
        segs.append(f"Q {x + 5} 1, {x + 10} 6 T {x + 20} 6")
        x += 20
    return svg_img(SQUIGGLE.format(w=width, path=" ".join(segs)), width, 10, "squiggle")


def svg_img(svg: str, width: int | None = None, height: int | None = None, cls: str = "") -> str:
    """Embed an SVG as a data-URI <img> (st.html's sanitizer strips inline <svg>)."""
    if "xmlns" not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    size = (f' width="{width}"' if width else "") + (f' height="{height}"' if height else "")
    return f'<img src="{uri}"{size} class="{cls}" alt="">'


def icon_img(name: str, size: int = 40) -> str:
    return svg_img(ICONS[name], size, size)


def leaf(width: int = 90) -> str:
    return svg_img(LEAF, width, int(width * 4 / 3))


def esc(text) -> str:
    return html.escape(str(text))


# ---------------------------------------------------------------- headings
def page_header(title: str, subtitle: str | None = None, icon: str | None = None) -> None:
    doodle = f'<div class="doodle">{icon_img(icon, 54)}</div>' if icon else ""
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.html(f'<div class="page-title"><div><h1>{esc(title)}</h1>{_squiggle(min(60 + 13 * len(title), 420))}</div>{doodle}</div>{sub}')


def section(title: str) -> None:
    st.html(f'<div class="section-title">{esc(title)}</div>{_squiggle(120)}')


# ---------------------------------------------------------------- cards/notes
def feature_cards(items: list[tuple[str, str, str, str]]) -> None:
    """items: (icon, title, description, colour in blue|sage|butter|pink)."""
    cards = "".join(
        f'<div class="feature bg-{c}">{icon_img(i)}<div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>'
        for i, t, d, c in items
    )
    st.html(f'<div class="feature-grid">{cards}</div>')


_NOTE_ICON = {"info": "💡", "warn": "⚠️", "ok": "✔️", "blue": "ℹ️"}


def note(text: str, kind: str = "info") -> None:
    """Pastel note. kind: info (yellow), warn (pink), ok (sage), blue."""
    st.html(f'<div class="note {kind}"><span class="ic">{_NOTE_ICON[kind]}</span><div>{text}</div></div>')


def sticky(text: str, color: str = "") -> None:
    st.html(f'<div style="text-align:center"><span class="sticky {color}">{esc(text)}</span></div>')


def pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill {kind}">{esc(text)}</span>'


def card(inner_html: str, title: str | None = None) -> None:
    t = f'<div class="card-title">{esc(title)}</div>' if title else ""
    st.html(f'<div class="card">{t}{inner_html}</div>')


def table(rows: list[tuple], header: tuple | None = None) -> str:
    """Return an HTML table. Numeric cells (int/float) are right-aligned."""
    def cell(v):
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            return f'<td class="num">{v:.4f}</td>' if isinstance(v, (float, np.floating)) else f'<td class="num">{v}</td>'
        return f"<td>{v}</td>"
    head = "<tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>" if header else ""
    body = "".join("<tr>" + "".join(cell(v) for v in r) + "</tr>" for r in rows)
    return f'<table class="tbl">{head}{body}</table>'


def show_table(rows, header=None) -> None:
    st.html(table(rows, header))


def pipeline(steps: list[tuple[str, str, str]], horizontal: bool = False) -> None:
    """steps: (title, description, colour)."""
    arrow = '<div class="pipe-arrow">→</div>' if horizontal else '<div class="pipe-arrow">↓</div>'
    parts = [f'<div class="pipe-step bg-{c}"><div class="t">{esc(t)}</div><div class="d">{esc(d)}</div></div>'
             for t, d, c in steps]
    st.html(f'<div class="pipeline{" h" if horizontal else ""}">{arrow.join(parts)}</div>')


def image_block(img, title: str, sub: str | None = None) -> None:
    """Titled image inside a bordered container."""
    with st.container(border=True):
        st.html(f'<div class="img-title">{esc(title)}</div>' + (f'<div class="img-sub">{esc(sub)}</div>' if sub else ""))
        st.image(img, width="stretch")


def heatmap_legend() -> None:
    st.html('<div class="legend"><span>0.0 · unlikely</span><div class="bar"></div><span>1.0 · likely tampered</span></div>')


def mask_legend() -> None:
    st.html('<div class="legend"><span><span class="sw" style="background:#fff"></span>predicted tampered</span>'
            '<span><span class="sw" style="background:#000"></span>not flagged</span></div>')
