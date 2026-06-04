from pathlib import Path

import streamlit as st

from config import APP_VERSION
from data.loader import discover
from data.loader import fasta as load_fasta_src
from state import AppState


def render(s: AppState) -> None:
    with st.sidebar:
        st.title("🧬 CAAT Visualizer")
        st.divider()

        st.subheader("📂 Primary Output")
        st.caption("Used by Mean Scores, Head Explorer, and 3D Structure.")
        _dir_picker(
            label="CAAT output folder",
            dir_attr="run_dir",
            s=s,
            on_update=s.update_files,
            on_reset=s.reset_files,
            show_full_summary=True,
        )

        st.divider()

        st.subheader("↔️ Difference Map Inputs")
        st.caption("Query and Target can be separate CAAT runs.")

        st.markdown("**Query**")
        _dir_picker(
            label="Query output folder",
            dir_attr="query_dir",
            s=s,
            on_update=s.update_query_files,
            on_reset=s.reset_query_files,
        )

        st.markdown("**Target**")
        _dir_picker(
            label="Target output folder",
            dir_attr="target_dir",
            s=s,
            on_update=s.update_target_files,
            on_reset=s.reset_target_files,
        )

        st.divider()

        st.subheader("🔤 Sequence (optional)")
        seq_src = st.radio(
            "Source", ["Paste", "FASTA file"], horizontal=True, key="_seq_source"
        )

        seq = ""
        if seq_src == "FASTA file" and s.fasta_files:
            chosen = st.selectbox(
                "FASTA", s.fasta_files, format_func=lambda f: f.name, key="_fasta_sel"
            )
            try:
                seq = load_fasta_src(chosen)
                st.caption(f"{len(seq)} residues loaded.")
            except Exception as e:
                st.error(f"Could not parse FASTA: {e}")
        else:
            seq = st.text_area(
                "Amino-acid sequence",
                value=s.sequence,
                height=80,
                placeholder="MGSSHHHHHHSSGLVPRGSHMLE…",
                key="_seq_paste",
            )

        if seq != s.sequence:
            s.sequence = seq
            s.save()

        st.divider()

        st.subheader("⚙️ Settings")
        pct = st.slider(
            "High-importance percentile",
            50,
            99,
            value=s.threshold_pct,
            key="_threshold",
        )
        top_n = st.number_input("Top-N residues", 1, 500, value=s.top_n, key="_top_n")

        if pct != s.threshold_pct or int(top_n) != s.top_n:
            s.threshold_pct = pct
            s.top_n = int(top_n)
            s.save()

        st.divider()
        st.caption(f"v{APP_VERSION}")


def _dir_picker(
    *,
    label: str,
    dir_attr: str,
    s: AppState,
    on_update,
    on_reset,
    show_full_summary: bool = False,
) -> None:
    current = getattr(s, dir_attr)
    run_dir = st.text_input(
        label,
        value=current,
        placeholder="/path/to/caat/outputs",
        key=f"_input_{dir_attr}",
    )

    if run_dir != current:
        setattr(s, dir_attr, run_dir)
        on_reset()

    if not run_dir:
        return

    p = Path(run_dir)
    if p.is_dir():
        files = discover(run_dir)
        on_update(files)
        npy = len(files["npy"])
        if npy == 0:
            st.warning("No .npy files found.")
        elif show_full_summary:
            st.success(
                f"**{npy}** .npy · "
                f"**{len(files['pdb'])}** PDB · "
                f"**{len(files['fasta'])}** FASTA"
            )
        else:
            st.success(f"**{npy}** .npy files found.")
    else:
        st.error("Directory not found.")
        on_reset()
