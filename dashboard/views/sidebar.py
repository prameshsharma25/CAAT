from pathlib import Path

import streamlit as st
from config import APP_VERSION
from data.loader import discover_npy, discover_viz
from state import AppState


def render(s: AppState) -> None:
    with st.sidebar:
        st.title("CAAT Visualizer")
        st.divider()

        st.subheader("🔍 Query")
        st.caption("Primary input — used by Mean Scores, Head Explorer, and Diff Map.")
        _npy_picker(
            label="Query output folder",
            dir_attr="query_dir",
            s=s,
            on_update=s.update_query,
            on_reset=s.reset_query,
        )

        st.divider()

        st.subheader("🎯 Target  *(optional)*")
        st.caption("When set, enables the Difference Map tab.")
        _npy_picker(
            label="Target output folder",
            dir_attr="target_dir",
            s=s,
            on_update=s.update_target,
            on_reset=s.reset_target,
        )

        st.divider()

        st.subheader("🗂️ Visualizations")
        st.caption(
            "Folder containing residue ranking CSVs and PDB files for 3D structure."
        )
        _viz_picker(s)

        st.divider()

        st.subheader("🔤 Sequence  *(optional)*")
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

        st.subheader("📐 Distogram Viewer  *(optional)*")
        st.caption("Folders for distogram .npz files and PDB structures.")
        _distogram_picker(s)
        st.caption(f"v{APP_VERSION}")


def _npy_picker(
    *,
    label: str,
    dir_attr: str,
    s: AppState,
    on_update,
    on_reset,
) -> None:
    current = getattr(s, dir_attr)
    path = st.text_input(
        label,
        value=current,
        placeholder="/path/to/caat/outputs",
        key=f"_input_{dir_attr}",
    )

    if path != current:
        setattr(s, dir_attr, path)
        on_reset()

    if not path:
        return

    p = Path(path)
    if p.is_dir():
        files = discover_npy(path)
        on_update(files)
        n = len(files["npy"])
        (
            st.success(f"**{n}** .npy file(s) found.")
            if n
            else st.warning("No .npy files found.")
        )
    else:
        st.error("Directory not found.")
        on_reset()


def _viz_picker(s: AppState) -> None:
    current = s.viz_dir
    path = st.text_input(
        "Visualizations folder",
        value=current,
        placeholder="/path/to/caat/visualizations",
        key="_input_viz_dir",
    )

    if path != current:
        s.viz_dir = path
        s.reset_viz()

    if not path:
        return

    p = Path(path)
    if p.is_dir():
        files = discover_viz(path)
        s.update_viz(files)
        csv_n = len(files["csv"])
        pdb_n = len(files["pdb"])
        if csv_n + pdb_n == 0:
            st.warning("No CSVs or PDBs found.")
        else:
            st.success(f"**{csv_n}** CSV · **{pdb_n}** PDB")
    else:
        st.error("Directory not found.")
        s.reset_viz()


def _distogram_picker(s: AppState) -> None:
    def _text(label, attr, placeholder):
        val = st.text_input(
            label, value=getattr(s, attr), placeholder=placeholder, key=f"_input_{attr}"
        )
        if val != getattr(s, attr):
            setattr(s, attr, val)
            s.save()

    _text("Protein A name", "distogram_name_a", "xcl1")
    _text(
        "Protein A distogram folder", "distogram_folder_a", "/path/to/xcl1_distograms"
    )
    _text(
        "Protein A structure folder",
        "distogram_structure_folder_a",
        "/path/to/xcl1_structures",
    )
    st.divider()
    _text("Protein B name", "distogram_name_b", "anc0")
    _text(
        "Protein B distogram folder", "distogram_folder_b", "/path/to/anc0_distograms"
    )
    _text(
        "Protein B structure folder",
        "distogram_structure_folder_b",
        "/path/to/anc0_structures",
    )
