from __future__ import annotations

import numpy as np
import pandas as pd


def to_float(arr: np.ndarray) -> np.ndarray:
    """
    Ensure arr is numeric float32.
    Handles ColabFold void dtypes (e.g. |V2 = float16 bytes).
    """
    if arr.dtype.kind in ("f", "i", "u"):
        return arr.astype(np.float32)
    reinterpret = {2: np.float16, 4: np.float32, 8: np.float64}.get(arr.dtype.itemsize)
    if reinterpret is None:
        raise ValueError(f"Cannot reinterpret dtype {arr.dtype} as float.")
    return arr.view(reinterpret).astype(np.float32)


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
    Trims to the shortest sequence length.
    """
    scores = [to_mean_scores(a) for a in arrays]
    n = min(len(s) for s in scores)
    return np.stack([s[:n] for s in scores]).mean(axis=0)


def residue_labels(seq: str, n: int) -> list[str]:
    """Return ['A1', 'C2', …] labels; fall back to plain indices if seq is short."""
    if seq and len(seq) >= n:
        return [f"{seq[i]}{i + 1}" for i in range(n)]
    return [str(i + 1) for i in range(n)]


def scores_from_csv(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a CAAT residue_ranking CSV.

    Returns
    -------
    res_nums : (N,) int array   — residue numbers in sequence order
    scores   : (N,) float array — attention scores aligned to res_nums
    """
    df = df[df["Amino acid"] != "-"].copy()
    df = df.sort_values("Residue number")
    res_nums = df["Residue number"].to_numpy(dtype=int)
    scores = df["Attention score"].to_numpy(dtype=float)
    return res_nums, scores


def merge_csv_scores(dfs: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """
    Average attention scores across multiple ranking CSVs.
    Aligns on residue number; only residues present in all files are kept.
    """
    parsed = [scores_from_csv(df) for df in dfs]
    common = set(parsed[0][0])
    for res_nums, _ in parsed[1:]:
        common &= set(res_nums)
    common = np.array(sorted(common), dtype=int)
    stacked = np.stack(
        [scores[np.isin(res_nums, common)] for res_nums, scores in parsed]
    )
    return common, stacked.mean(axis=0)


def inject_bfactor(
    pdb_text: str,
    res_nums: np.ndarray,
    scores: np.ndarray,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> str:
    """
    Rewrite the B-factor column using percentile-based normalisation so that
    colour variation isn't dominated by a small number of outlier residues.
    Scores are clipped to [low_pct, high_pct] percentile then scaled to 0–100.
    """
    lo = np.percentile(scores, low_pct)
    hi = np.percentile(scores, high_pct)
    norm = np.clip((scores - lo) / (hi - lo + 1e-9), 0.0, 1.0) * 100
    score_map = dict(zip(res_nums.tolist(), norm.tolist()))

    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                rn = int(line[22:26].strip())
                bf = score_map.get(rn, 0.0)
                line = line[:60] + f"{bf:6.2f}" + line[66:]
            except (ValueError, IndexError):
                pass
        lines.append(line)
    return "\n".join(lines)
