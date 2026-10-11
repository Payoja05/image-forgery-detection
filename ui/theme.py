"""Global CSS for the light, hand-made look of the design reference.

Base colours/fonts come from .streamlit/config.toml; this file styles the
custom HTML components in ui/components.py and a few Streamlit elements.
"""
import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&family=Nunito+Sans:wght@400;600;700&display=swap');
:root{
  --cream:#FAF7F0; --blush:#F4E6DF; --ink:#292929; --muted:#6F6B65;
  --blue:#DCEAF5; --sage:#E4EDDF; --butter:#FFF0C7; --pink:#F7E0DD; --border:#D8D0C7;
  --blue-ink:#4F7CA8; --sage-ink:#5E7F55; --pink-ink:#B5524A; --butter-ink:#8A6D1C;
  --paper:#FFFDF8;
  --hand:'Patrick Hand','Segoe Print','Comic Neue',cursive;
}
.stApp{
  background-color:var(--cream);
  /* faint paper grain */
  background-image:radial-gradient(rgba(120,100,80,.035) 1px, transparent 1px);
  background-size:4px 4px;
}
[data-testid="stMainBlockContainer"], .block-container{max-width:1180px; padding-top:2.2rem;}
[data-testid="stSidebar"]{border-right:1px solid var(--border);}
[data-testid="stSidebarHeader"]{height:.5rem; min-height:.5rem; padding:0;}
h1,h2,h3,h4{font-family:var(--hand) !important; color:var(--ink); font-weight:400 !important; letter-spacing:.2px;}

/* ---------- sidebar brand + nav ---------- */
.brand{display:flex; flex-direction:column; align-items:center; text-align:center; padding:.4rem 0 1rem;
  border-bottom:1px dashed var(--border); margin-bottom:.8rem;}
.brand svg{width:34px; height:34px;}
.brand .name{font-family:var(--hand); font-size:1.18rem; line-height:1.2; margin-top:.3rem; color:var(--ink);}
.nav-active{display:flex; align-items:center; gap:.55rem; background:var(--blue); border:1px solid #C9DCEC;
  border-radius:6px; padding:.42rem .7rem; margin:.1rem 0 .25rem; font-size:.95rem; font-weight:600; color:var(--ink);}
.nav-active .material{font-family:'Material Symbols Rounded'; font-size:1.15rem; color:var(--blue-ink);}
[data-testid="stSidebar"] [data-testid="stPageLink"] a{border-radius:6px; padding:.3rem .7rem;}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{background:rgba(220,234,245,.6);}
.sidebar-leaf{margin-top:2.5rem; opacity:.75; display:flex; justify-content:center;}

/* ---------- headings with hand-drawn underline ---------- */
.page-title{display:flex; align-items:flex-start; gap:1rem; margin-bottom:.4rem;}
.page-title h1{font-size:2.5rem; line-height:1.15; margin:0; padding:0;}
.page-title .doodle svg{width:54px; height:54px;}
.squiggle{display:block; margin:.15rem 0 .2rem;}
.subtitle{color:var(--muted); font-size:1.02rem; margin:.3rem 0 1.2rem; max-width:720px;}
.section-title{font-family:var(--hand); font-size:1.55rem; margin:1.4rem 0 .3rem; color:var(--ink);}
.hand{font-family:var(--hand);}

/* ---------- cards ---------- */
.card{background:var(--paper); border:1px solid var(--border); border-radius:6px; padding:1rem 1.1rem;}
.card-title{font-weight:700; font-size:.98rem; margin-bottom:.35rem;}
.feature-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.9rem; margin:.4rem 0 1rem;}
.feature{border:1px solid rgba(0,0,0,.06); border-radius:6px; padding:1.1rem .9rem; text-align:center; min-height:150px;}
.feature svg{width:40px; height:40px; margin-bottom:.5rem;}
.feature .t{font-weight:700; font-size:.95rem; margin-bottom:.2rem;}
.feature .d{font-size:.86rem; color:var(--muted); line-height:1.35;}
.bg-blue{background:var(--blue);} .bg-sage{background:var(--sage);} .bg-butter{background:var(--butter);} .bg-pink{background:var(--pink);}

/* ---------- notes / status ---------- */
.note{border-radius:6px; padding:.7rem .95rem; margin:.5rem 0 .9rem; font-size:.93rem; line-height:1.45;
  border:1px solid rgba(0,0,0,.06); display:flex; gap:.6rem; align-items:flex-start;}
.note .ic{font-size:1.05rem; line-height:1.3;}
.note.info{background:var(--butter);} .note.warn{background:var(--pink);} .note.ok{background:var(--sage);} .note.blue{background:var(--blue);}
.sticky{font-family:var(--hand); font-size:1.08rem; line-height:1.25; background:var(--butter); color:#4a4030;
  padding:.65rem .8rem; border-radius:3px; box-shadow:1px 2px 0 rgba(0,0,0,.07); transform:rotate(-2deg);
  display:inline-block; max-width:220px; margin:.4rem .2rem;}
.sticky.pink{background:var(--pink); transform:rotate(1.5deg);} .sticky.blue{background:var(--blue); transform:rotate(-1deg);}
.sticky.sage{background:var(--sage); transform:rotate(2deg);}
.pill{display:inline-block; border-radius:4px; padding:.18rem .6rem; font-weight:700; font-size:.95rem; border:1px solid rgba(0,0,0,.08);}
.pill.warn{background:var(--pink); color:var(--pink-ink);} .pill.ok{background:var(--sage); color:var(--sage-ink);}
.pill.neutral{background:var(--blue); color:var(--blue-ink);}
.caption-hand{font-family:var(--hand); color:var(--muted); font-size:1.02rem; text-align:center; margin:.25rem 0 .1rem;}
.img-title{font-weight:700; font-size:.93rem; margin:0 0 .35rem; text-align:center;}
.img-sub{font-size:.8rem; color:var(--muted); text-align:center; margin-top:-.3rem; margin-bottom:.35rem;}

/* ---------- tables ---------- */
.tbl{width:100%; border-collapse:collapse; font-size:.92rem; background:var(--paper);}
.tbl td,.tbl th{border:1px solid var(--border); padding:.45rem .65rem; text-align:left;}
.tbl th{background:#F3EDE4; font-weight:700;}
.tbl td.num{text-align:right; font-variant-numeric:tabular-nums;}

/* ---------- vertical pipeline ---------- */
.pipeline{display:flex; flex-direction:column; align-items:center; gap:0;}
.pipe-step{width:min(320px,100%); text-align:center; border:1px solid rgba(0,0,0,.08); border-radius:6px; padding:.55rem .8rem;}
.pipe-step .t{font-weight:700; font-size:.92rem;} .pipe-step .d{font-size:.78rem; color:var(--muted);}
.pipe-arrow{color:var(--muted); font-size:1.1rem; line-height:1.3;}
.pipeline.h{flex-direction:row; flex-wrap:wrap; justify-content:center; gap:.3rem;}
.pipeline.h .pipe-step{width:auto; min-width:110px;}
.pipeline.h .pipe-arrow{align-self:center;}

/* ---------- heatmap legend ---------- */
.legend{display:flex; align-items:center; gap:.5rem; font-size:.8rem; color:var(--muted); margin:.3rem 0;}
.legend .bar{flex:1; height:10px; border-radius:2px; border:1px solid var(--border);
  background:linear-gradient(90deg,#00008f,#0000ff,#00ffff,#ffff00,#ff0000,#800000);}
.legend .sw{display:inline-block; width:14px; height:14px; border:1px solid var(--border); vertical-align:middle; margin-right:.25rem;}

/* ---------- Streamlit widgets ---------- */
[data-testid="stFileUploaderDropzone"]{border:2px dashed #B9AEA2 !important; background:var(--paper) !important; padding:2.2rem 1rem !important;}
.stButton>button[kind="primary"], .stDownloadButton>button{border-radius:6px;}
.stButton>button[kind="primary"]{background:var(--blue); color:var(--ink); border:1px solid #9DBBD6; font-weight:700;}
.stButton>button[kind="primary"]:hover{background:#cfe2f1; border-color:var(--blue-ink); color:var(--ink);}
[data-testid="stImageCaption"]{font-family:var(--hand); font-size:1rem; color:var(--muted);}
[data-testid="stExpander"] details{background:var(--paper);}
footer{visibility:hidden;}
</style>
"""


def apply_theme() -> None:
    st.html(CSS)
