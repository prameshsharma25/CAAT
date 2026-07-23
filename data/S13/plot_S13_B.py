import pandas as pd
import re
import sys
import numpy as np
from matplotlib import pyplot as plt


def gini(x):
    x = np.array(x, dtype=float)

    if np.amin(x) < 0:
        x = x - np.amin(x)

    x = x + 1e-12
    x = np.sort(x)

    n = len(x)
    index = np.arange(1, n + 1)

    return np.sum((2 * index - n - 1) * x) / (n * np.sum(x))


def gini_claude(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values."""
    if not values:
        raise ValueError("List must not be empty.")

    n = len(values)
    sorted_vals = sorted(values)
    total = sum(sorted_vals)

    if total == 0:
        return 0.0

    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def parse_protein(protein_str):
    """
    Parses strings like '1ABC.A123B' into:
    PDB_ID = 1ABC
    mutation = A123B
    """
    pdb_id, mut = protein_str.split(".")
    return pdb_id, mut


def organize_by_pdb(df):
    """
    Organizes dataframe into:
    {
      PDB_ID: {
          mutation: {
              "delta_pLDDT": [...],
              "delta_pLDDT_median": ...,
              "attention": [...],
              "ddG": [...]
          }
      }
    }
    """

    df[["PDB_ID", "Residue"]] = df["Protein"].apply(
        lambda x: pd.Series(parse_protein(x))
    )

    result = {}
    ids = []

    for pdb_id, group in df.groupby("PDB_ID"):
        if pdb_id not in ids:
            ids.append(pdb_id)

        residue_dict = {}

        for _, row in group.iterrows():
            res = row["Residue"]

            if res not in residue_dict:
                residue_dict[res] = {
                    "delta_pLDDT": [],
                    "delta_pLDDT_median": -999,
                    "attention": [],
                    "ddG": [],
                }

            residue_dict[res]["delta_pLDDT"].append(row["delta pLDDT"])
            residue_dict[res]["attention"].append(row["Normalized attention"])
            residue_dict[res]["ddG"].append(abs(row["ddG"]))

        result[pdb_id] = residue_dict

    for pdb_id, residues in result.items():
        for res, values in residues.items():
            values["delta_pLDDT_median"] = abs(np.median(values["delta_pLDDT"]))

    print(ids)
    print(len(ids))

    return result


def decile_fractions_same_order(dplddt, ddg):
    """
    Compute decile contributions using the ΔpLDDT ranking.

    dplddt and ddg must already be sorted in ΔpLDDT order.
    """

    deciles = pd.qcut(np.arange(len(dplddt)), q=10, labels=False)

    af_fracs = []
    exp_fracs = []

    for d in range(10):
        mask = deciles == d

        af_fracs.append(
            dplddt[mask].sum() / dplddt.sum() * 100
        )

        exp_fracs.append(
            ddg[mask].sum() / ddg.sum() * 100
        )

    return np.array(af_fracs), np.array(exp_fracs)


def make_fig(dplddt, ddg):
    af_fracs, exp_fracs = decile_fractions_same_order(dplddt, ddg)

    # Keeping these annotation positions the same as your ΔTM figure.
    L39E_af_decile = 10
    L39E_exp_decile = 2

    fig, ax = plt.subplots(figsize=(7, 3.5))

    deciles = np.arange(1, 11)
    width = 0.35

    BLACK = "#222222"
    RED = "#AE0639"

    ax.bar(
        deciles - width / 2,
        af_fracs,
        width,
        color=BLACK,
        label=f"AlphaFold ΔpLDDT  (Gini = 0.43)",
        zorder=3,
    )

    ax.bar(
        deciles + width / 2,
        exp_fracs,
        width,
        color=RED,
        label=f"Experimental ΔΔG  (Gini = 0.32)",
        zorder=3,
    )

    ax.axhline(
        10,
        color="#aaa",
        linewidth=0.8,
        linestyle="--",
        zorder=2,
        label="Uniform distribution (10%)",
    )

    ax.set_xlabel(
        "Residue sensitivity decile (1=least, 10=most)",
        fontsize=11,
    )
    ax.set_ylabel("% of total sensitivity", fontsize=11)
    ax.set_title("", fontsize=12, fontweight="bold")

    ax.set_xticks(deciles)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=9)

    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig("./S13_plddt.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    df = pd.read_csv('./all_rocklin_mutated_updated_TM.csv')

    result = organize_by_pdb(df)

    median_pLDDTs = {}
    res_ddGs = {}

    L39E_dPLDDT = -999

    for pdb_id, residues in result.items():
        for res, values in residues.items():

            key = pdb_id + "." + res[1:-1]

            if key not in median_pLDDTs:
                median_pLDDTs[key] = []
            median_pLDDTs[key].append(values["delta_pLDDT_median"])

            if key not in res_ddGs:
                res_ddGs[key] = []
            res_ddGs[key].append(values["ddG"])

            if pdb_id + "." + res == "6SOW.L39E":
                L39E_dPLDDT = values["delta_pLDDT_median"]

    top_AVG = []
    SOW_39_val = -999

    for pdb_id, pLDDTs in median_pLDDTs.items():
        top_AVG.append(np.mean(sorted(pLDDTs)[-3:]))

        if pdb_id == "6SOW.39":
            print(top_AVG[-1])
            SOW_39_val = top_AVG[-1]

    print(SOW_39_val, SOW_39_val / max(top_AVG))
    print(gini(top_AVG))
    print(gini_claude(top_AVG))

    dplddt = np.array(top_AVG)

    ddG_AVG = []

    for pdb_id, ddGs in res_ddGs.items():
        ddGs_sorted = sorted([x[0] for x in ddGs])
        ddG_AVG.append(np.mean(ddGs_sorted[-3:]))

    ddg = np.array(ddG_AVG)

    # Sort both arrays using only the ΔpLDDT order.
    order = np.argsort(dplddt)

    final_AVGs = dplddt[order]
    final_ddGs = ddg[order]

    SOW_39_IDX = np.where(final_AVGs <= L39E_dPLDDT)[0][-1]
    x_final = np.arange(1, len(final_AVGs) + 1) / len(final_AVGs)

    SOW_39_dG_IDX = np.where(final_ddGs < 0.6)[0][-1]
    ddg_x_final = np.arange(1, len(final_ddGs) + 1) / len(final_ddGs)

    print(SOW_39_dG_IDX, final_ddGs[SOW_39_dG_IDX])

    make_fig(np.array(final_AVGs), np.array(final_ddGs))