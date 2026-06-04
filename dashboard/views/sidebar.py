from pathlib import Path

import streamlit as st

from config import APP_VERSION
from data.loader import categorise_uploads, discover
from data.loader import fasta as load_fasta_src
from state import AppState


def render(s: AppState) -> None:
    """Render the sidebar. Mutates and saves `s` in place."""
    with st.sidebar:
        st.title("🧬 CAAT Visualizer")
        st.caption("Post-processing dashboard — no GPU required.")
        st.divider()

        mode = st.radio(
            "Load files via",
            ["Directory path", "File upload"],
            horizontal=True,
            key="_load_mode",
        )

        st.subheader("📂 Primary Output")
        st.caption("Used by all tabs except Difference Maps.")
        if mode == "Directory path":
            _dir_picker(
                s,
                dir_key="run_dir",
                label="CAAT output folder",
                on_update=s.update_files,
                on_reset=s.reset_files,
            )
        else:
            _uploader(
                s,
                upload_key="_primary_upload",
                on_update=s.update_files,
                on_reset=s.reset_files,
            )

        st.divider()

        st.subheader("↔️ Difference Map Inputs")
        st.caption("Query and Target can be separate CAAT runs.")

        st.markdown("**Query**")
        if mode == "Directory path":
            _dir_picker(
                s,
                dir_key="query_dir",
                label="Query output folder",
                on_update=s.update_query_files,
                on_reset=s.reset_query_files,
                npy_count_attr="query_npy_files",
            )
        else:
            _uploader(
                s,
                upload_key="_query_upload",
                on_update=s.update_query_files,
                on_reset=s.reset_query_files,
                label="Query .npy files",
            )

        st.markdown("**Target**")
        if mode == "Directory path":
            _dir_picker(
                s,
                dir_key="target_dir",
                label="Target output folder",
                on_update=s.update_target_files,
                on_reset=s.reset_target_files,
                npy_count_attr="target_npy_files",
            )
        else:
            _uploader(
                s,
                upload_key="_target_upload",
                on_update=s.update_target_files,
                on_reset=s.reset_target_files,
                label="Target .npy files",
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

        st.subheader("⚙️  Settings")
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
    s: AppState,
    *,
    dir_key: str,
    label: str,
    on_update,
    on_reset,
    npy_count_attr: str = "npy_files",
) -> None:
    current = getattr(s, dir_key)
    run_dir = st.text_input(
        label,
        value=current,
        placeholder="/path/to/caat/outputs",
        key=f"_input_{dir_key}",
    )

    if run_dir != current:
        setattr(s, dir_key, run_dir)
        on_reset()

    if not run_dir:
        return

    p = Path(run_dir)
    if p.is_dir():
        files = discover(run_dir)
        on_update(files)
        npy_count = len(files["npy"])
        if npy_count == 0:
            st.warning("No .npy files found.")
        else:
            extra = ""
            if npy_count_attr == "npy_files":
                extra = (
                    f" · **{len(files['img'])}** images"
                    f" · **{len(files['pdb'])}** PDB"
                    f" · **{len(files['fasta'])}** FASTA"
                )
            st.success(f"**{npy_count}** .npy{extra}")
    else:
        st.error("Directory not found.")
        on_reset()


def _uploader(
    s: AppState,
    *,
    upload_key: str,
    on_update,
    on_reset,
    label: str = "Select all files from a CAAT run",
) -> None:
    uploaded = st.file_uploader(
        label,
        type=["npy", "pdb", "fasta", "fa", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=upload_key,
        help="Hold Cmd/Ctrl to select multiple files at once.",
    )

    if not uploaded:
        st.info("Upload .npy tensors (and optionally PDB, FASTA, images).")
        on_reset()
        return

    files = categorise_uploads(uploaded)
    on_update(files)
    npy_count = len(files["npy"])
    st.success(f"**{npy_count}** .npy loaded.")
