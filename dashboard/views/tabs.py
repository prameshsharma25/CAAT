from __future__ import annotations

from itertools import combinations

import data.loader as loader
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from config import DIVERGING_SCALE, GRADIENT_CMAP, HEATMAP_SCALE, MAX_HEADS_PER_ROW
from processing.attention import (
    diff_map,
    inject_bfactor,
    mean_scores_across,
    merge_csv_scores,
    residue_labels,
    scores_from_csv,
    to_float,
    to_mean_scores,
)
from state import AppState

try:
    import py3Dmol

    HAS_3D = True
except ImportError:
    HAS_3D = False


def _xaxis_tick_config(indices: np.ndarray, labels: list[str], max_ticks: int = 20) -> dict:
    n = len(indices)
    if n <= max_ticks:
        return dict(tickmode="array", tickvals=indices.tolist(), ticktext=labels)
    step = max(1, int(np.ceil(n / max_ticks)))
    tickvals = indices[::step].tolist()
    ticktext = [labels[i] for i in range(0, n, step)]
    return dict(tickmode="array", tickvals=tickvals, ticktext=ticktext)


def render(s: AppState) -> None:
    st.subheader("Per-Residue Mean Attention Score")
    st.caption(
        "Averaged over all layers, heads, and query positions from the raw tensor."
    )

    if not s.has_query:
        st.info("Set a Query output folder in the sidebar.")
        return

    chosen = st.selectbox(
        "Attention tensor (.npy)",
        s.query_npy_files,
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
    residue_indices = np.arange(1, n + 1)
    xaxis_config = _xaxis_tick_config(residue_indices, labels)

    col_bar, col_line = st.columns(2)

    with col_bar:
        colors = ["crimson" if v >= cutoff else "steelblue" for v in scores]
        fig = go.Figure(
            go.Bar(
                x=residue_indices,
                y=scores,
                customdata=labels,
                marker_color=colors,
                hovertemplate="<b>%{customdata}</b><br>Score: %{y:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Mean Attention  (red ≥ {s.threshold_pct}th percentile)",
            xaxis_title="Residue",
            yaxis_title="Score",
            xaxis=xaxis_config,
            height=380,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_line:
        fig2 = go.Figure(
            go.Scatter(
                x=residue_indices,
                y=scores,
                mode="lines+markers",
                customdata=labels,
                hovertemplate="<b>%{customdata}</b><br>Score: %{y:.5f}<extra></extra>",
            )
        )
        fig2.add_hline(
            y=cutoff,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{s.threshold_pct}th pct",
        )
        fig2.update_layout(
            title="Score Profile",
            xaxis_title="Residue",
            yaxis_title="Score",
            xaxis=xaxis_config,
            xaxis_rangeslider_visible=True,
            height=380,
            margin=dict(t=40, b=10),
        )
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

    if not s.has_query:
        st.info("Set a Query output folder in the sidebar.")
        return

    chosen = st.selectbox(
        "Attention tensor (.npy)",
        s.query_npy_files,
        format_func=lambda f: f.name,
        key="head_npy_sel",
    )
    arr = to_float(loader.npy(chosen))

    if arr.ndim != 4:
        st.warning(f"Expected shape (L, H, N, N) but got `{arr.shape}`.")
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
        "Computes residue×residue delta (Query − Target) from raw attention tensors. "
        "Set separate Query and Target folders in the sidebar."
    )

    if not s.diff_ready:
        st.info("Set both a Query and a Target output folder in the sidebar.")
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
    residue_indices = np.arange(1, N + 1)
    xaxis_config = _xaxis_tick_config(residue_indices, labels)

    fig = px.imshow(
        delta,
        x=residue_indices,
        y=residue_indices,
        color_continuous_scale=DIVERGING_SCALE,
        zmin=-abs_max,
        zmax=abs_max,
        labels=dict(color="Δ"),
        title=f"Δ Attention  |  {q_file.name}  −  {t_file.name}",
        aspect="auto",
    )
    fig.update_traces(hovertemplate="Residue %{x} × %{y}<br>Δ %{z:.5f}<extra></extra>")
    fig.update_layout(height=540, xaxis=xaxis_config, yaxis=xaxis_config)
    st.plotly_chart(fig, use_container_width=True)

    proj = delta.mean(axis=1)
    fig2 = go.Figure(
        go.Bar(
            x=residue_indices,
            y=proj,
            customdata=labels,
            marker_color=["crimson" if v > 0 else "steelblue" for v in proj],
            hovertemplate="<b>%{customdata}</b><br>Avg Δ: %{y:.5f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title="Row-mean projection  (positive = Query > Target)",
        xaxis_title="Residue",
        yaxis_title="Avg Δ Score",
        xaxis=xaxis_config,
        xaxis_rangeslider_visible=True,
        height=320,
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_diff_csv(s: AppState) -> None:
    st.subheader("Difference CSV Plot")
    st.caption(
        "Plot per-residue attention-difference CSVs saved from the analysis pipeline."
    )

    if not s.csv_files:
        st.info("Set a Visualizations folder containing CSV files in the sidebar.")
        return

    chosen_csv = st.selectbox(
        "Difference CSV",
        s.csv_files,
        format_func=lambda f: f.name,
        key="diff_csv_sel",
    )

    df = loader.csv(chosen_csv)
    required_cols = {"Residue number", "Attention difference"}
    if not required_cols.issubset(df.columns):
        st.warning(
            "Selected CSV is not a valid attention-difference file. "
            "It must contain at least 'Residue number' and 'Attention difference' columns."
        )
        return

    residue_numbers = df["Residue number"].astype(int)
    full_residue_numbers = np.arange(residue_numbers.min(), residue_numbers.max() + 1)

    if "Amino acid" not in df.columns:
        df["Amino acid"] = ""
    df["Amino acid"] = df["Amino acid"].fillna("").astype(str)
    df["Attention difference"] = df["Attention difference"].astype(float)

    if "Attention difference negative-only" in df.columns:
        df["Attention difference negative-only"] = df[
            "Attention difference negative-only"
        ].astype(float)
    else:
        df["Attention difference negative-only"] = np.clip(
            df["Attention difference"].to_numpy(), None, 0
        )

    full_df = (
        pd.DataFrame({"Residue number": full_residue_numbers})
        .merge(df, how="left", on="Residue number")
        .sort_values("Residue number")
    )
    full_df["Amino acid"] = full_df["Amino acid"].fillna("").astype(str)
    full_df["Attention difference"] = (
        full_df["Attention difference"].fillna(0.0).astype(float)
    )
    full_df["Attention difference negative-only"] = (
        full_df["Attention difference negative-only"].fillna(0.0).astype(float)
    )

    residue_numbers = full_df["Residue number"].to_numpy()
    amino_acids = full_df["Amino acid"].tolist()
    labels = [
        f"{aa}{rn}" if aa else str(rn) for rn, aa in zip(residue_numbers, amino_acids)
    ]

    raw_diff = full_df["Attention difference"].to_numpy()
    negative_only = full_df["Attention difference negative-only"].to_numpy()
    residue_indices = full_df["Residue number"].to_numpy()
    xaxis_config = _xaxis_tick_config(residue_indices, labels)

    plot_mode = st.radio(
        "Plot mode",
        ["All residues", "Negative-only"],
        index=0,
        key="diff_csv_plot_mode",
    )
    show_all_residues = plot_mode == "All residues"
    y_values = raw_diff if show_all_residues else negative_only
    bar_title = (
        "Attention Difference (all residues)"
        if show_all_residues
        else "Attention Difference (negative-only)"
    )

    col_bar, col_line = st.columns(2)
    with col_bar:
        fig = go.Figure(
            go.Bar(
                x=residue_indices,
                y=y_values,
                customdata=labels,
                marker_color=[
                    "crimson" if v > 0 else "royalblue" if v < 0 else "lightgray"
                    for v in y_values
                ],
                hovertemplate="<b>%{customdata}</b><br>Δ: %{y:.5f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=bar_title,
            xaxis_title="Residue",
            yaxis_title="Δ Attention",
            xaxis=xaxis_config,
            xaxis_rangeslider_visible=True,
            height=420,
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_line:
        fig2 = go.Figure(
            go.Scatter(
                x=residue_indices,
                y=raw_diff,
                mode="lines+markers",
                customdata=labels,
                line=dict(color="#1f77b4"),
                hovertemplate="<b>%{customdata}</b><br>Δ: %{y:.5f}<extra></extra>",
            )
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            title="Raw Attention Difference",
            xaxis_title="Residue",
            yaxis_title="Δ Attention",
            xaxis=xaxis_config,
            xaxis_rangeslider_visible=True,
            height=420,
            margin=dict(t=40, b=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    table_cols = ["Residue number", "Amino acid", "Attention difference"]
    if "Attention difference negative-only" in df.columns:
        table_cols.append("Attention difference negative-only")

    top_n = int(min(s.top_n, len(df)))
    top_df = (
        df.assign(**{"Attention difference negative-only": negative_only})
        .sort_values("Attention difference negative-only")
        .head(top_n)
        .reset_index(drop=True)
    )

    st.subheader(f"Top-{top_n} Decreasing Residues")
    st.dataframe(top_df[table_cols], use_container_width=True)


def render_3d(s: AppState) -> None:
    st.subheader("3D Structure — Residues Colored by Attention Score")

    if not HAS_3D:
        st.warning("Install py3Dmol to enable this tab:  `pip install py3Dmol`")
        return

    if not s.pdb_files:
        st.info("No .pdb files found in the selected directory.")
        return

    chosen_pdb = st.selectbox(
        "PDB file", s.pdb_files, format_func=lambda f: f.name, key="3d_pdb_sel"
    )
    pdb_text = loader.pdb(chosen_pdb)
    view = py3Dmol.view(width=720, height=520)

    if not s.viz_ready:
        view.addModel(pdb_text, "pdb")
        view.setStyle({}, {"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        components.html(view._make_html(), height=520)
        st.caption(
            "Spectrum coloring. Add a Visualizations folder with CSVs to color by attention score."
        )
        return

    rank_csv_files = [f for f in s.csv_files if f.name.endswith("_residue_ranking.csv")]
    if not rank_csv_files:
        st.info(
            "No residue-ranking CSVs found. Add `*_residue_ranking.csv` files to the Visualizations folder."
        )
        return

    filter_csv = st.toggle("Filter CSVs to average", value=False, key="3d_csv_filter")
    selected_csv = (
        st.multiselect(
            "Select CSVs to average",
            options=rank_csv_files,
            default=rank_csv_files,
            format_func=lambda f: f.name,
            key="3d_csv_sel",
        )
        if filter_csv
        else rank_csv_files
    )

    if not selected_csv:
        st.info("Select at least one CSV.")
        return

    with st.spinner(f"Averaging scores across {len(selected_csv)} CSV(s)…"):
        dfs = [loader.csv(f) for f in selected_csv]
        res_nums, scores = merge_csv_scores(dfs)

    col_lo, col_hi = st.columns(2)
    low_pct = col_lo.slider("Color floor (percentile)", 0, 49, 5, key="3d_lo_pct")
    high_pct = col_hi.slider("Color ceiling (percentile)", 51, 100, 95, key="3d_hi_pct")

    pdb_mod = inject_bfactor(pdb_text, res_nums, scores, low_pct, high_pct)
    view.addModel(pdb_mod, "pdb")
    view.setStyle({}, {"cartoon": {"colorscheme": {"prop": "b", "gradient": "rwb"}}})
    view.zoomTo()
    components.html(view._make_html(), height=520)
    st.caption(
        f"Averaged across **{len(selected_csv)}** CSV(s).  "
        "🔵 Low attention  →  🔴 High attention"
    )
