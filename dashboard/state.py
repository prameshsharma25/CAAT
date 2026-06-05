from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

_SESSION_KEY = "_caat_state"


@dataclass
class AppState:
    query_dir: str = ""
    query_npy_files: list = field(default_factory=list)
    target_dir: str = ""
    target_npy_files: list = field(default_factory=list)
    viz_dir: str = ""
    csv_files: list = field(default_factory=list)
    pdb_files: list = field(default_factory=list)
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

    def reset_query(self) -> None:
        self.query_npy_files = []
        self.save()

    def reset_target(self) -> None:
        self.target_npy_files = []
        self.save()

    def reset_viz(self) -> None:
        self.csv_files = []
        self.pdb_files = []
        self.save()

    def update_query(self, files: dict[str, list]) -> None:
        self.query_npy_files = files.get("npy", [])
        self.save()

    def update_target(self, files: dict[str, list]) -> None:
        self.target_npy_files = files.get("npy", [])
        self.save()

    def update_viz(self, files: dict[str, list]) -> None:
        self.csv_files = files.get("csv", [])
        self.pdb_files = files.get("pdb", [])
        self.save()

    @property
    def has_query(self) -> bool:
        return bool(self.query_npy_files)

    @property
    def diff_ready(self) -> bool:
        return bool(self.query_npy_files and self.target_npy_files)

    @property
    def viz_ready(self) -> bool:
        return bool(self.csv_files and self.pdb_files)

    def __repr__(self) -> str:
        return (
            f"AppState("
            f"query_dir={self.query_dir!r}, query_npy={len(self.query_npy_files)}, "
            f"target_dir={self.target_dir!r}, target_npy={len(self.target_npy_files)}, "
            f"viz_dir={self.viz_dir!r}, csv={len(self.csv_files)}, pdb={len(self.pdb_files)}, "
            f"threshold={self.threshold_pct}, top_n={self.top_n})"
        )
