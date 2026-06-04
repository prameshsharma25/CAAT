from __future__ import annotations

import numpy as np


def to_mean_scores(arr: np.ndarray) -> np.ndarray:
    """
    Collapse a raw attention tensor to a per-residue 1-D mean score.

    Shapes handled
    --------------
    (L, H, N, N)  ColabFold raw output  → mean over L & H, then query axis
    (N, N)         pre-averaged 2-D map  → mean over rows
    (N,)           already 1-D           → returned as-is
    """
    match arr.ndim:
        case 4:
            return arr.mean(axis=(0, 1)).mean(axis=0)
        case 2:
            return arr.mean(axis=0)
        case _:
            return arr.squeeze()


def to_avg_map(arr: np.ndarray) -> np.ndarray:
    """
    Return a 2-D (N, N) attention map averaged over layers and heads.
    Accepts (L, H, N, N) or (N, N).
    """
    match arr.ndim:
        case 4:
            return arr.mean(axis=(0, 1))
        case 2:
            return arr
        case _:
            raise ValueError(f"Cannot convert shape {arr.shape} to 2-D map.")


def diff_map(arr_q: np.ndarray, arr_t: np.ndarray) -> np.ndarray:
    """
    Compute residue×residue delta (Query − Target).
    Trims to the shorter sequence length if they differ.
    """
    map_q = to_avg_map(arr_q)
    map_t = to_avg_map(arr_t)
    n = min(map_q.shape[0], map_t.shape[0])
    return map_q[:n, :n] - map_t[:n, :n]


def normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-9)


def residue_labels(seq: str, n: int) -> list[str]:
    """Return ['A1', 'C2', …] labels; fall back to plain indices if seq is short."""
    if seq and len(seq) >= n:
        return [f"{seq[i]}{i + 1}" for i in range(n)]
    return [str(i + 1) for i in range(n)]


def inject_bfactor(pdb_text: str, norm_scores: np.ndarray) -> str:
    """
    Rewrite the B-factor column of a PDB string with normalised attention
    scores (scaled 0–100) for py3Dmol gradient colouring.
    """
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                ri = int(line[22:26].strip()) - 1
                bf = norm_scores[ri] * 100 if ri < len(norm_scores) else 0.0
                line = line[:60] + f"{bf:6.2f}" + line[66:]
            except (ValueError, IndexError):
                pass
        lines.append(line)
    return "\n".join(lines)
