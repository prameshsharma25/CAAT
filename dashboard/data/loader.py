from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import streamlit as st
from config import IMAGE_EXTS

try:
    from Bio import SeqIO

    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# Accepts a filesystem path or a Streamlit UploadedFile
FileSource = Union[str, Path, "st.runtime.uploaded_file_manager.UploadedFile"]


def discover(run_dir: str) -> dict[str, list[Path]]:
    """Walk run_dir and return categorised Path lists."""
    p = Path(run_dir)
    if not p.is_dir():
        return {k: [] for k in ("npy", "img", "pdb", "fasta")}
    return {
        "npy": sorted(p.glob("*.npy")),
        "img": sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS),
        "pdb": sorted(p.glob("*.pdb")),
        "fasta": sorted(p.glob("*.fasta")) + sorted(p.glob("*.fa")),
    }


def categorise_uploads(uploaded_files: list) -> dict[str, list]:
    """Sort UploadedFile objects into the same shape as discover()."""
    out: dict[str, list] = {"npy": [], "img": [], "pdb": [], "fasta": []}
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        if ext == ".npy":
            out["npy"].append(f)
        elif ext in IMAGE_EXTS:
            out["img"].append(f)
        elif ext == ".pdb":
            out["pdb"].append(f)
        elif ext in {".fasta", ".fa"}:
            out["fasta"].append(f)
    for k in out:
        out[k].sort(key=lambda f: f.name)
    return out


def _is_uploaded(src: FileSource) -> bool:
    return not isinstance(src, (str, Path))


def _cache_key(src: FileSource) -> str:
    return src.name if _is_uploaded(src) else str(src)


def _read_bytes(src: FileSource) -> bytes:
    if _is_uploaded(src):
        src.seek(0)
        return src.read()
    return Path(src).read_bytes()


def _read_text(src: FileSource) -> str:
    if _is_uploaded(src):
        src.seek(0)
        return src.read().decode("utf-8", errors="replace")
    return Path(src).read_text()


@st.cache_data(show_spinner="Loading tensor…")
def _load_npy(key: str, _src: FileSource) -> np.ndarray:
    return np.load(io.BytesIO(_read_bytes(_src)))


@st.cache_data(show_spinner="Reading PDB…")
def _load_pdb(key: str, _src: FileSource) -> str:
    return _read_text(_src)


@st.cache_data(show_spinner="Parsing sequence…")
def _load_fasta(key: str, _src: FileSource) -> str:
    text = _read_text(_src)
    if HAS_BIO:
        record = next(SeqIO.parse(io.StringIO(text), "fasta"))
        return str(record.seq)
    return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))


def npy(src: FileSource) -> np.ndarray:
    return _load_npy(_cache_key(src), src)


def pdb(src: FileSource) -> str:
    return _load_pdb(_cache_key(src), src)


def fasta(src: FileSource) -> str:
    return _load_fasta(_cache_key(src), src)
