from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import data.loader as loader
from config import (
    DEFAULT_COLS,
    DIVERGING_SCALE,
    GRADIENT_CMAP,
    HEATMAP_SCALE,
    MAX_HEADS_PER_ROW,
)
from processing.attention import (
    diff_map,
    inject_bfactor,
    normalise,
    residue_labels,
    to_mean_scores,
)
from state import AppState

try:
    import py3Dmol

    HAS_3D = True
except ImportError:
    HAS_3D = False


def render(s: AppState) -> None:
    st.subheader("Per-Residue Mean Attention Score")
    st.caption(
        "Averaged over all layers, heads, and query positions from the raw tensor."
    )

    if not s.npy_files:
        st.info("Point the sidebar to a CAAT output directory containing .npy files.")
        return

    chosen = st.selectbox(
        "Attention tensor (.npy)",
        s.npy_files,
        format_func=lambda f: f.name,
        key="mean_npy_sel",
    )

    with st.spinner("Computing mean scores…"):
        arr = loader.npy(chosen)

    st.caption(f"Shape: `{arr.shape}`  dtype: `{arr.dtype}`")

    scores = to_mean_scores(arr)
    n = len(scores)
    labels = residue_labels(s.sequence, n)
    cutoff = float(np.percentile(scores, s.threshold_pct))

    col_bar, col_line = st.columns(2)

    with col_bar:
        colors = ["crimson" if v >= cutoff else "steelblue" for v in scores]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=scores,
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Score: %{y:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Mean Attention  (red ≥ {s.threshold_pct}th percentile)",
            xaxis_title="Residue",
            yaxis_title="Score",
            height=380,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_line:
        fig2 = px.line(
            x=labels,
            y=scores,
            labels={"x": "Residue", "y": "Score"},
            title="Score Profile",
        )
        fig2.add_hline(
            y=cutoff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{s.threshold_pct}th pct",
        )
        fig2.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(f"Top-{s.top_n} Residues")
    df = (
        pd.DataFrame(
            {
                "Residue": labels,
                "Index": range(1, n + 1),
                "Mean Attention": scores,
            }
        )
        .sort_values("Mean Attention", ascending=False)
        .reset_index(drop=True)
        .assign(**{"High Importance": lambda d: d["Mean Attention"] >= cutoff})
    )
    st.dataframe(
        df.head(s.top_n).style.background_gradient(
            subset=["Mean Attention"], cmap=GRADIENT_CMAP
        ),
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download full rankings CSV",
        df.to_csv(index=False),
        file_name=f"{chosen.name.rsplit('.', 1)[0]}_rankings.csv",
        mime="text/csv",
    )


def render_heads(s: AppState) -> None:
    st.subheader("Attention Head Explorer")
    st.caption("Inspect individual attention heads from a raw (L × H × N × N) tensor.")

    if not s.npy_files:
        st.info("No .npy files found.")
        return

    chosen = st.selectbox(
        "Attention tensor (.npy)",
        s.npy_files,
        format_func=lambda f: f.name,
        key="head_npy_sel",
    )
    arr = loader.npy(chosen)

    if arr.ndim != 4:
        st.warning(
            f"Expected shape (L, H, N, N) but got `{arr.shape}`. "
            "This tab requires a raw ColabFold attention tensor."
        )
        return

    L, H, N, _ = arr.shape
    labels = residue_labels(s.sequence, N)

    col_l, col_h = st.columns(2)
    layer_i = col_l.slider("Layer", 0, L - 1, 0, key="head_layer")
    head_i = col_h.slider("Head", 0, H - 1, 0, key="head_head")

    fig = px.imshow(
        arr[layer_i, head_i],
        x=labels,
        y=labels,
        color_continuous_scale=HEATMAP_SCALE,
        labels=dict(color="Attn"),
        title=f"Layer {layer_i}  ·  Head {head_i}",
        aspect="auto",
    )
    fig.update_layout(height=540)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"🔲 All {H} heads — Layer {layer_i}"):
        cols = st.columns(min(H, MAX_HEADS_PER_ROW))
        for h in range(H):
            mini = px.imshow(
                arr[layer_i, h],
                color_continuous_scale=HEATMAP_SCALE,
                title=f"Head {h}",
                aspect="auto",
            )
            mini.update_layout(
                height=180, margin=dict(t=28, b=4, l=4, r=4), coloraxis_showscale=False
            )
            mini.update_xaxes(visible=False)
            mini.update_yaxes(visible=False)
            cols[h % MAX_HEADS_PER_ROW].plotly_chart(mini, use_container_width=True)


def render_diff(s: AppState) -> None:
    st.subheader("Difference Map: Query vs. Target")
    st.caption(
        "Loads .npy files from the Query and Target directories set in the sidebar. "
        "The dashboard computes the residue×residue delta (Query − Target) from "
        "their layer/head-averaged attention maps."
    )

    if not s.diff_ready:
        st.info(
            "Set both a **Query** and a **Target** output folder in the sidebar "
            "to compute a difference map."
        )
        return

    col_q, col_t = st.columns(2)
    q_file = col_q.selectbox(
        "Query .npy", s.query_npy_files, format_func=lambda f: f.name, key="diff_q_sel"
    )
    t_file = col_t.selectbox(
        "Target .npy",
        s.target_npy_files,
        format_func=lambda f: f.name,
        key="diff_t_sel",
    )

    with st.spinner("Computing difference map…"):
        delta = diff_map(loader.npy(q_file), loader.npy(t_file))

    N = delta.shape[0]
    labels = residue_labels(s.sequence, N)
    abs_max = float(np.abs(delta).max()) or 1.0

    fig = px.imshow(
        delta,
        x=labels,
        y=labels,
        color_continuous_scale=DIVERGING_SCALE,
        zmin=-abs_max,
        zmax=abs_max,
        labels=dict(color="Δ"),
        title=f"Δ Attention  |  {q_file.name}  −  {t_file.name}",
        aspect="auto",
    )
    fig.update_layout(height=540)
    st.plotly_chart(fig, use_container_width=True)

    proj = delta.mean(axis=1)
    fig2 = go.Figure(
        go.Bar(
            x=labels,
            y=proj,
            marker_color=["crimson" if v > 0 else "steelblue" for v in proj],
            hovertemplate="<b>%{x}</b><br>Avg Δ: %{y:.5f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title="Row-mean projection  (positive = Query > Target)",
        xaxis_title="Residue",
        yaxis_title="Avg Δ Score",
        height=320,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_compare(s: AppState) -> None:
    st.subheader("Multi-Protein Mean Score Comparison")
    st.caption("Overlay and diff mean attention profiles across multiple .npy files.")

    if len(s.npy_files) < 2:
        st.info("Load a directory with at least 2 .npy files to compare.")
        return

    selected = st.multiselect(
        "Select proteins to overlay",
        options=s.npy_files,
        default=s.npy_files[: min(3, len(s.npy_files))],
        format_func=lambda f: f.name,
        key="compare_sel",
    )

    if not selected:
        return

    score_map: dict[str, np.ndarray] = {
        f.name: to_mean_scores(loader.npy(f)) for f in selected
    }

    fig = go.Figure(
        [
            go.Scatter(
                x=list(range(1, len(s) + 1)),
                y=s,
                mode="lines",
                name=name,
                hovertemplate=f"<b>{name}</b><br>Res %{{x}}<br>Score: %{{y:.5f}}<extra></extra>",
            )
            for name, s in score_map.items()
        ]
    )
    fig.update_layout(
        title="Overlaid Mean Attention Profiles",
        xaxis_title="Residue Index",
        yaxis_title="Mean Score",
        height=420,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(selected) >= 2:
        st.subheader("Pairwise Delta")
        pairs = list(combinations(selected, 2))
        pair = st.selectbox(
            "Pair",
            pairs,
            format_func=lambda p: f"{p[0].name}  −  {p[1].name}",
            key="compare_pair_sel",
        )
        s0, s1 = score_map[pair[0].name], score_map[pair[1].name]
        mn = min(len(s0), len(s1))
        d = s0[:mn] - s1[:mn]

        fig2 = go.Figure(
            go.Bar(
                x=list(range(1, mn + 1)),
                y=d,
                marker_color=["crimson" if v > 0 else "steelblue" for v in d],
                hovertemplate="Res %{x}<br>Δ: %{y:.5f}<extra></extra>",
            )
        )
        fig2.update_layout(
            title=f"{pair[0].name}  −  {pair[1].name}",
            xaxis_title="Residue Index",
            yaxis_title="Δ Score",
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_gallery(s: AppState) -> None:
    st.subheader("Pre-Rendered CAAT Graphs")
    st.caption("Mean-attention and difference-map images already saved by CAAT.")

    if not s.img_files:
        st.info("No image files found in the selected directory.")
        return

    kw = st.text_input(
        "Filter by filename",
        placeholder="mean  /  diff  /  protein name…",
        key="gallery_filter",
    )
    filtered = (
        [f for f in s.img_files if kw.lower() in f.name.lower()] if kw else s.img_files
    )

    if not filtered:
        st.warning("No images match that filter.")
        return

    n_cols = st.slider("Columns", 1, 4, DEFAULT_COLS, key="gallery_cols")
    cols = st.columns(n_cols)
    for i, img in enumerate(sorted(filtered, key=lambda f: f.name)):
        cols[i % n_cols].image(str(img), caption=img.name, use_container_width=True)


def render_3d(s: AppState) -> None:
    st.subheader("3D Structure — Residues Colored by Attention Score")

    if not HAS_3D:
        st.warning("Install py3Dmol to enable this tab:  `pip install py3Dmol`")
        return

    if not s.pdb_files:
        st.info("No .pdb files found in the selected directory.")
        return

    col_p, col_s = st.columns(2)
    chosen_pdb = col_p.selectbox(
        "PDB file", s.pdb_files, format_func=lambda f: f.name, key="3d_pdb_sel"
    )

    score_options = ["None (color by spectrum)"] + [f.name for f in s.npy_files]
    chosen_score_name = col_s.selectbox(
        "Attention tensor (.npy) — optional", score_options, key="3d_npy_sel"
    )

    pdb_text = loader.pdb(chosen_pdb)
    view = py3Dmol.view(width=720, height=520)

    if chosen_score_name == "None (color by spectrum)":
        view.addModel(pdb_text, "pdb")
        view.setStyle({}, {"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        components.html(view._make_html(), height=520)
        st.caption(
            "Spectrum coloring. Select a .npy file to color residues by attention score."
        )
    else:
        npy_src = next(f for f in s.npy_files if f.name == chosen_score_name)
        scores = to_mean_scores(loader.npy(npy_src))
        norm = normalise(scores)
        pdb_mod = inject_bfactor(pdb_text, norm)

        view.addModel(pdb_mod, "pdb")
        view.setStyle(
            {}, {"cartoon": {"colorscheme": {"prop": "b", "gradient": "rwb"}}}
        )
        view.zoomTo()
        components.html(view._make_html(), height=520)
        st.caption("🔵 Low attention  →  🔴 High attention  (scores in B-factor column).")
