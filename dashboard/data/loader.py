from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from config import IMAGE_EXTS

try:
    from Bio import SeqIO

    HAS_BIO = True
except ImportError:
    HAS_BIO = False


def discover(run_dir: str) -> dict[str, list[Path]]:
    """Walk run_dir and return categorised Path lists."""
    p = Path(run_dir)
    if not p.is_dir():
        return {k: [] for k in ("npy", "pdb", "fasta")}
    return {
        "npy": sorted(p.glob("*.npy")),
        "pdb": sorted(p.glob("*.pdb")),
        "fasta": sorted(p.glob("*.fasta")) + sorted(p.glob("*.fa")),
    }


@st.cache_data(show_spinner="Loading tensor…")
def _load_npy(path: str) -> np.ndarray:
    return np.load(path)


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


def pdb(src: Path) -> str:
    return _load_pdb(str(src))


def fasta(src: Path) -> str:
    return _load_fasta(str(src))
