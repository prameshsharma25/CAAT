import streamlit as st
from config import APP_TITLE
from state import AppState
from views.sidebar import render as render_sidebar
from views.tabs import render as render_mean
from views.tabs import render_diff, render_heads, render_diff_csv, render_distogram

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

s = AppState.load()
render_sidebar(s)

tab_mean, tab_heads, tab_diff, tab_diff_csv, tab_distogram = st.tabs(
    [
        "📊 Mean Scores",
        "🔬 Head Explorer",
        "↔️ Difference Maps",
        "📈 Difference CSV",
        "📐 Distogram Viewer",
    ]
)

with tab_mean:
    render_mean(s)
with tab_heads:
    render_heads(s)
with tab_diff:
    render_diff(s)
with tab_diff_csv:
    render_diff_csv(s)
with tab_distogram:
    render_distogram(s)