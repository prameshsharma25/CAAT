from pathlib import Path

import streamlit as st
from config import APP_VERSION
from data.loader import categorise_uploads, discover
from data.loader import fasta as load_fasta_src
from state import AppState


def render(s: AppState) -> None:
    """Render the sidebar. Mutates and saves `s` in place."""
    with st.sidebar:
        st.title("CAAT Visualizer")
        st.divider()

        st.subheader("📂 Load Files")
        mode = st.radio(
            "Source",
            ["Directory path", "File upload"],
            horizontal=True,
            key="_load_mode",
        )

        if mode == "Directory path":
            _dir_picker(s)
        else:
            _uploader(s)

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


def _dir_picker(s: AppState) -> None:
    run_dir = st.text_input(
        "Path to CAAT output folder",
        value=s.run_dir,
        placeholder="/path/to/caat/outputs",
        key="_run_dir_input",
    )

    if run_dir != s.run_dir:
        s.run_dir = run_dir
        s.reset_files()

    if not run_dir:
        return

    p = Path(run_dir)
    if p.is_dir():
        files = discover(run_dir)
        s.update_files(files)
        counts = (
            len(files["npy"]),
            len(files["img"]),
            len(files["pdb"]),
            len(files["fasta"]),
        )
        if sum(counts) == 0:
            st.warning("Directory found but no recognised files.")
        else:
            npy, imgs, pdbs, fas = counts
            st.success(
                f"**{npy}** .npy · **{imgs}** images · "
                f"**{pdbs}** PDB · **{fas}** FASTA"
            )
    else:
        st.error("Directory not found.")
        s.reset_files()


def _uploader(s: AppState) -> None:
    uploaded = st.file_uploader(
        "Select all files from a CAAT run",
        type=["npy", "pdb", "fasta", "fa", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="_file_uploader",
        help="Hold Cmd/Ctrl to select multiple files at once.",
    )

    if not uploaded:
        st.info("Upload .npy tensors, PDB, FASTA, and/or graph images.")
        s.reset_files()
        return

    files = categorise_uploads(uploaded)
    s.update_files(files)
    npy, imgs, pdbs, fas = (
        len(files["npy"]),
        len(files["img"]),
        len(files["pdb"]),
        len(files["fasta"]),
    )
    st.success(
        f"**{npy}** .npy · **{imgs}** images · " f"**{pdbs}** PDB · **{fas}** FASTA"
    )
