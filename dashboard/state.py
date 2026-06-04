from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

_SESSION_KEY = "_caat_state"


@dataclass
class AppState:
    run_dir: str = ""
    npy_files: list = field(default_factory=list)
    pdb_files: list = field(default_factory=list)
    fasta_files: list = field(default_factory=list)

    query_dir: str = ""
    target_dir: str = ""
    query_npy_files: list = field(default_factory=list)
    target_npy_files: list = field(default_factory=list)

    sequence: str = ""
    threshold_pct: int = 90
    top_n: int = 10

    @classmethod
    def load(cls) -> "AppState":
        if _SESSION_KEY not in st.session_state:
            st.session_state[_SESSION_KEY] = cls()
        return st.session_state[_SESSION_KEY]

    def save(self) -> None:
        st.session_state[_SESSION_KEY] = self

    def reset_files(self) -> None:
        self.npy_files = []
        self.pdb_files = []
        self.fasta_files = []
        self.save()

    def reset_query_files(self) -> None:
        self.query_npy_files = []
        self.save()

    def reset_target_files(self) -> None:
        self.target_npy_files = []
        self.save()

    def update_files(self, files: dict[str, list]) -> None:
        self.npy_files = files.get("npy", [])
        self.pdb_files = files.get("pdb", [])
        self.fasta_files = files.get("fasta", [])
        self.save()

    def update_query_files(self, files: dict[str, list]) -> None:
        self.query_npy_files = files.get("npy", [])
        self.save()

    def update_target_files(self, files: dict[str, list]) -> None:
        self.target_npy_files = files.get("npy", [])
        self.save()

    @property
    def has_files(self) -> bool:
        return bool(self.npy_files or self.pdb_files)

    @property
    def diff_ready(self) -> bool:
        return bool(self.query_npy_files and self.target_npy_files)

    def __repr__(self) -> str:
        return (
            f"AppState("
            f"run_dir={self.run_dir!r}, npy={len(self.npy_files)}, "
            f"pdb={len(self.pdb_files)}, fasta={len(self.fasta_files)}, "
            f"query_dir={self.query_dir!r}, query_npy={len(self.query_npy_files)}, "
            f"target_dir={self.target_dir!r}, target_npy={len(self.target_npy_files)}, "
            f"threshold={self.threshold_pct}, top_n={self.top_n})"
        )
