import streamlit as st
from config import APP_TITLE
from state import AppState
from views.sidebar import render as render_sidebar
from views.tabs import render as render_mean
from views.tabs import (
    render_3d,
    render_compare,
    render_diff,
    render_gallery,
    render_heads,
)

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

s = AppState.load()

render_sidebar(s)

tab_mean, tab_heads, tab_diff, tab_compare, tab_gallery, tab_3d = st.tabs(
    [
        "📊 Mean Scores",
        "🔬 Head Explorer",
        "↔️ Difference Maps",
        "🆚 Multi-Protein Compare",
        "🖼️ Saved Graphs",
        "🧱 3D Structure",
    ]
)

with tab_mean:
    render_mean(s)
with tab_heads:
    render_heads(s)
with tab_diff:
    render_diff(s)
with tab_compare:
    render_compare(s)
with tab_gallery:
    render_gallery(s)
with tab_3d:
    render_3d(s)
