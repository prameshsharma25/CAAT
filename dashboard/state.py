from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import streamlit as st

_SESSION_KEY = "_caat_state"


@dataclass
class AppState:
    run_dir: str = ""
    npy_files: list = field(default_factory=list)
    img_files: list = field(default_factory=list)
    pdb_files: list = field(default_factory=list)
    fasta_files: list = field(default_factory=list)

    sequence: str = ""

    threshold_pct: int = 90
    top_n: int = 10

    @classmethod
    def load(cls) -> "AppState":
        """Return the existing AppState from session_state, or create a fresh one."""
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = cls()
        return st.session_state[_SESSION_KEY]

    def save(self) -> None:
        """Persist this instance back into session_state (no-op if already same object)."""
        st.session_state[_SESSION_KEY] = self

    def reset_files(self) -> None:
        """Clear all file lists (called when the output directory changes)."""
        self.npy_files = []
        self.img_files = []
        self.pdb_files = []
        self.fasta_files = []
        self.save()

    def update_files(self, files: dict[str, list]) -> None:
        """Bulk-update file lists from a discover() / categorise_uploads() result."""
        self.npy_files = files.get("npy", [])
        self.img_files = files.get("img", [])
        self.pdb_files = files.get("pdb", [])
        self.fasta_files = files.get("fasta", [])
        self.save()

    @property
    def has_files(self) -> bool:
        return bool(self.npy_files or self.img_files or self.pdb_files)

    def __repr__(self) -> str:
        return (
            f"AppState(run_dir={self.run_dir!r}, "
            f"npy={len(self.npy_files)}, img={len(self.img_files)}, "
            f"pdb={len(self.pdb_files)}, fasta={len(self.fasta_files)}, "
            f"threshold={self.threshold_pct}, top_n={self.top_n})"
        )
