from __future__ import annotations

import numpy as np


def to_float(arr: np.ndarray) -> np.ndarray:
    """
    Ensure arr is a numeric float32 array.
    ColabFold sometimes saves tensors as void/structured dtype (e.g. |V2);
    we reinterpret the raw bytes as float16 then upcast to float32.
    """
    if arr.dtype.kind in ("f", "i", "u"):  # already numeric
        return arr.astype(np.float32)
    # Void dtype — reinterpret bytes. |V2 = 2 bytes → float16
    byte_width = arr.dtype.itemsize
    reinterpret_dtype = {2: np.float16, 4: np.float32, 8: np.float64}.get(byte_width)
    if reinterpret_dtype is None:
        raise ValueError(
            f"Unrecognised dtype {arr.dtype} with itemsize {byte_width}. "
            "Cannot reinterpret as a float type."
        )
    return arr.view(reinterpret_dtype).astype(np.float32)


def to_mean_scores(arr: np.ndarray) -> np.ndarray:
    """
    Collapse a raw attention tensor to a per-residue 1-D mean score.

    Shapes handled
    --------------
    (L, H, N, N)  ColabFold raw output  → mean over L & H, then query axis
    (N, N)         pre-averaged 2-D map  → mean over rows
    (N,)           already 1-D           → returned as-is
    """
    arr = to_float(arr)
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
    arr = to_float(arr)
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


def mean_scores_across(arrays: list[np.ndarray]) -> np.ndarray:
    """
    Average per-residue mean scores across multiple attention tensors.
    Tensors may differ in sequence length — trims to the shortest.
    """
    scores = [to_mean_scores(a) for a in arrays]
    n = min(len(s) for s in scores)
    return np.stack([s[:n] for s in scores]).mean(axis=0)


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

    Builds a mapping from PDB residue number → sequential index so that
    non-contiguous or non-zero-based residue numbering is handled correctly.
    """
    res_nums: list[int] = []
    seen: set[int] = set()
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                rn = int(line[22:26].strip())
                if rn not in seen:
                    res_nums.append(rn)
                    seen.add(rn)
            except ValueError:
                pass

    res_to_idx = {rn: i for i, rn in enumerate(res_nums)}

    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                rn = int(line[22:26].strip())
                idx = res_to_idx.get(rn, -1)
                bf = norm_scores[idx] * 100 if 0 <= idx < len(norm_scores) else 0.0
                line = line[:60] + f"{bf:6.2f}" + line[66:]
            except (ValueError, IndexError):
                pass
        lines.append(line)
    return "\n".join(lines)
