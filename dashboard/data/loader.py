from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from Bio import SeqIO

    HAS_BIO = True
except ImportError:
    HAS_BIO = False


def discover_npy(run_dir: str) -> dict[str, list[Path]]:
    """Discover .npy files from a query or target directory."""
    p = Path(run_dir)
    if not p.is_dir():
        return {"npy": []}
    return {"npy": sorted(p.glob("*.npy"))}


def discover_viz(viz_dir: str) -> dict[str, list[Path]]:
    """Discover CSVs and PDBs from the visualizations directory."""
    p = Path(viz_dir)
    if not p.is_dir():
        return {"csv": [], "pdb": []}
    return {
        "csv": sorted(p.glob("*_residue_ranking.csv")),
        "pdb": sorted(p.glob("*.pdb")),
    }


@st.cache_data(show_spinner="Loading tensor…")
def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


@st.cache_data(show_spinner="Loading ranking CSV…")
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Reading PDB…")
def _load_pdb(path: str) -> str:
    return Path(path).read_text()


@st.cache_data(show_spinner="Parsing sequence…")
def _load_fasta(path: str) -> str:
    text = Path(path).read_text()
    if HAS_BIO:
        import io

        record = next(SeqIO.parse(io.StringIO(text), "fasta"))
        return str(record.seq)
    return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))


def npy(src: Path) -> np.ndarray:
    return _load_npy(str(src))


def csv(src: Path) -> pd.DataFrame:
    return _load_csv(str(src))


def pdb(src: Path) -> str:
    return _load_pdb(str(src))


def fasta(src: Path) -> str:
    return _load_fasta(str(src))
