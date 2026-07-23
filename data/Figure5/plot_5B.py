import pandas as pd
import re, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import random

def gini(x):
    x = np.array(x, dtype=float)
    
    if np.amin(x) < 0:
        x = x - np.amin(x)  # shift to non-negative
    
    x = x + 1e-12  # avoid division issues
    x = np.sort(x)
    
    n = len(x)
    index = np.arange(1, n + 1)
    
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def gini_claude(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of non-negative values."""
    if not values:
        raise ValueError("List must not be empty.")
    n = len(values)
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0  # Perfect equality (all zeros)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def bootstrap_gini(
    dist_a: list[float],
    dist_b: list[float],
    n_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals for two Gini coefficients and
    test whether their difference is statistically significant.

    Returns a dict with CIs for each distribution and for the difference,
    plus a p-value for the null hypothesis that the two Ginis are equal.
    """
    rng = random.Random(seed)

    def resample(data):
        return [rng.choice(data) for _ in data]

    ginis_a, ginis_b, diffs = [], [], []
    for _ in range(n_iterations):
        ga = gini(resample(dist_a))
        gb = gini(resample(dist_b))
        ginis_a.append(ga)
        ginis_b.append(gb)
        diffs.append(ga - gb)

    def ci(samples, confidence):
        lo = (1 - confidence) / 2
        hi = 1 - lo
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return sorted_s[int(lo * n)], sorted_s[int(hi * n)]

    observed_diff = gini(dist_a) - gini(dist_b)
    # Two-sided p-value: proportion of bootstrap diffs on the opposite side of zero
    p_value = sum(1 for d in diffs if (d <= 0 if observed_diff > 0 else d >= 0)) / n_iterations
    p_value = 2 * p_value  # two-sided

    alpha = 1 - confidence
    return {
        "gini_a":          gini(dist_a),
        "gini_b":          gini(dist_b),
        "ci_a":            ci(ginis_a, confidence),
        "ci_b":            ci(ginis_b, confidence),
        "observed_diff":   observed_diff,
        "ci_diff":         ci(diffs, confidence),
        "p_value":         min(p_value, 1.0),
        "n_iterations":    n_iterations,
        "confidence":      confidence,
    }


def print_results(r: dict) -> None:
    pct = int(r["confidence"] * 100)
    print(f"{'='*50}")
    print(f"  Bootstrap Gini Comparison ({r['n_iterations']:,} iterations)")
    print(f"{'='*50}")
    print(f"  Gini A: {r['gini_a']:.4f}  {pct}% CI: [{r['ci_a'][0]:.4f}, {r['ci_a'][1]:.4f}]")
    print(f"  Gini B: {r['gini_b']:.4f}  {pct}% CI: [{r['ci_b'][0]:.4f}, {r['ci_b'][1]:.4f}]")
    print(f"  Difference (A - B): {r['observed_diff']:.4f}  "
          f"{pct}% CI: [{r['ci_diff'][0]:.4f}, {r['ci_diff'][1]:.4f}]")
    print(f"  P-value (two-sided): {r['p_value']:.4f}")
    sig = r['p_value'] < (1 - r['confidence'])
    print(f"  Significant at {pct}% level: {'Yes' if sig else 'No'}")
    print(f"{'='*50}")


def parse_protein(protein_str):
    """
    Parses strings like '1ABC.A123B' into:
    PDB_ID = 1ABC
    residue_number = 123
    """
    pdb_id, mut = protein_str.split(".")

    return pdb_id, mut

    match = re.match(r"([A-Z])(\d+)([A-Z])", mut)
    if match:
        original_aa, res_num, mutated_aa = match.groups()
        return pdb_id, int(res_num)
    else:
        return pdb_id, None


def organize_by_pdb(df):
    """
    Organizes dataframe into:
    {
      PDB_ID: {
          residue_number: {
              "delta_pLDDT": [...],
              "attention": [...]
          }
      }
    }
    """
    # Parse Protein column
    df[["PDB_ID", "Residue"]] = df["Protein"].apply(
        lambda x: pd.Series(parse_protein(x))
    )

    #df.to_csv('test.csv')

    result = {}

    ids = []

    for pdb_id, group in df.groupby("PDB_ID"):
        if pdb_id not in ids:
            ids.append(pdb_id)
        residue_dict = {}

        #if pdb_id == '6SOW':
            #group.to_csv('test2.csv')

        for _, row in group.iterrows():
            res = row["Residue"]

            if res not in residue_dict:
                residue_dict[res] = {
                    "delta_pLDDT": [],
                    "attention": [],
                    "ddG":[],
                    "delta_TM": [],
                    "delta_TM_median": -999,
                }

            residue_dict[res]["delta_pLDDT"].append(row["delta pLDDT"])
            residue_dict[res]["attention"].append(row["Normalized attention"])
            residue_dict[res]["ddG"].append(abs(row["ddG"]))
            residue_dict[res]["delta_TM"].append(row["delta TM"])

        result[pdb_id] = residue_dict

    for pdb_id, residues in result.items():
        for res, values in residues.items():
            if np.median(values["delta_TM"]) <= -0.3:
            #if len([x for x in values["delta_TM"] if x <= -0.3]) == len(values["delta_TM"]):
                print(pdb_id+'.'+res,values["attention"][0],np.median(values["delta_TM"]))
            values["delta_TM_median"] = abs(np.median(values["delta_TM"]))

    print(ids)
    print(len(ids))

    return result

def print_result(result):
    
    print("\n=== Organized Output ===")
    for pdb_id, residues in result.items():
        print(f"\nPDB ID: {pdb_id}")
        for res, values in residues.items():
            print(f"  Residue {res}:")
            print(f"    delta_pLDDT: {values['delta_pLDDT']}")
            print(f"    attention:   {values['attention']}")

# ── Compute per-decile % of total for each metric ──────────────────────────────
def decile_fractions(values):
    """Return % of total absolute value in each decile (1=lowest, 10=highest)."""
    ranks = pd.Series(values).rank(method="first")
    deciles = pd.qcut(ranks, q=10, labels=False)  # 0-9
    total = values.sum()
    fracs = []
    for d in range(10):
        frac = values[deciles == d].sum() / total * 100
        fracs.append(frac)
    return np.array(fracs)

def make_fig(dtm,ddg):

    af_fracs  = decile_fractions(dtm)
    exp_fracs = decile_fractions(ddg)

    # ── L39E decile positions ──────────────────────────────────────────────────────
    # Based on confirmed values:
    # L39E is in the 95th percentile for AF → decile 10 (index 9)
    # L39E is in the 15th percentile for experimental → decile 2 (index 1)
    L39E_af_decile  = 10   # 1-indexed
    L39E_exp_decile = 2    # 1-indexed

    # ── Plot ───────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    deciles = np.arange(1, 11)
    width   = 0.35
    BLACK   = "#222222"
    RED     = "#AE0639"
    GOLD    = "#CA9823"
    
    bars_af  = ax.bar(deciles - width/2, af_fracs,  width, 
                      color=BLACK, label=f"AlphaFold ΔTM  (Gini = 0.43)",
                      zorder=3)
    bars_exp = ax.bar(deciles + width/2, exp_fracs, width,
                      color=RED,   label=f"Experimental ΔΔG  (Gini = 0.32)",
                      zorder=3)

    # ── L39E annotations ───────────────────────────────────────────────────────────
    # Arrow on AF bar (decile 10)
    af_bar_height = af_fracs[L39E_af_decile - 1]
    ax.annotate("Protein A L39E\n(AF)",
                xy=(L39E_af_decile - width/2, af_bar_height),
                xytext=(L39E_af_decile - width/2 + 0.5, af_bar_height +3),
                fontsize=8, color=BLACK, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2))
    
    # Arrow on experimental bar (decile 2)
    exp_bar_height = exp_fracs[L39E_exp_decile - 1]
    ax.annotate("Protein A L39E\n(Experimental\nΔΔG = −0.6)",
                xy=(L39E_exp_decile + width/2, exp_bar_height),
                xytext=(L39E_exp_decile + width/2 + 1, exp_bar_height + 2),
                fontsize=8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    
    # ── Reference line at uniform distribution (10% per decile) ───────────────────
    ax.axhline(10, color="#aaa", linewidth=0.8, linestyle="--", zorder=2,
               label="Uniform distribution (10%)")
    
    # ── Labels & style ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Residue sensitivity decile (1=least, 10=most)",
                  fontsize=11)
    ax.set_ylabel("% of total sensitivity", fontsize=11)
    ax.set_xticks(deciles)
    ax.set_xticklabels([str(d) for d in deciles], fontsize=9)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    
    plt.tight_layout()
    plt.savefig("fig5A.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    df = pd.read_csv('./all_rocklin_mutated_updated_TM.csv')


    # ---- Run organization ----
    result = organize_by_pdb(df)

    attention_high = []
    attention_low = []
    dplDDT_max_low = []
    dplDDT_max_high = []
    dTM_max_high = []
    destabilizing_burden_high = []
    destabilizing_burden_low = []
    plddt_changing_high = []
    plddt_changing_low = []
    median_TMs = {}
    res_ddGs = {}

    L39E_dTM = -999

    for pdb_id, residues in result.items():
        
        
        for res, values in residues.items():
            if pdb_id+'.'+res[1:-1] not in median_TMs:
                median_TMs[pdb_id+'.'+res[1:-1]] = []
            median_TMs[pdb_id+'.'+res[1:-1]].append(values["delta_TM_median"])
            if pdb_id+'.'+res[1:-1] not in res_ddGs:
                res_ddGs[pdb_id+'.'+res[1:-1]] = []
            res_ddGs[pdb_id+'.'+res[1:-1]].append(values["ddG"])

            if pdb_id+'.'+res == '6SOW.L39E':
                L39E_dTM=values["delta_TM_median"]

            

    top_AVG = []
    SOW_39_val = -999
    for pdb_id, TMs in median_TMs.items():
        top_AVG.append(np.mean(sorted(TMs)[-3:]))
        if pdb_id == '6SOW.39':
            #print(top_AVG[-1])
            SOW_39_val=top_AVG[-1]

    
    # print(SOW_39_val,SOW_39_val/max(top_AVG))
    # print(gini(top_AVG))
    # print(gini_claude(top_AVG))

    final_AVGs = sorted(top_AVG)
    SOW_39_IDX = np.where(final_AVGs <= L39E_dTM)[0][-1]
    x_final = np.arange(1,len(final_AVGs)+1)/len(final_AVGs)

    ddG_AVG = []
    SOW_39_ddG = -999
    for pdb_id, ddGs in res_ddGs.items():
        ddGs_sorted = sorted([x[0] for x in ddGs])
        ddG_AVG.append(np.mean(ddGs_sorted[-3:]))
        if pdb_id == '6SOW.39':
            #print(ddG_AVG[-1])
            SOW_39_ddG=ddG_AVG[-1]


    # print(gini(ddG_AVG))
    # print(gini_claude(ddG_AVG))

    final_ddGs = sorted(ddG_AVG)


    ddG_AVGs = np.array(sorted(ddG_AVG))
    SOW_39_dG_IDX = np.where(ddG_AVGs<0.6)[0][-1]
    ddg_x_final = np.arange(1,len(ddG_AVGs)+1)/len(ddG_AVGs)

    #print(SOW_39_dG_IDX,ddG_AVGs[SOW_39_dG_IDX])

    make_fig(np.array(final_AVGs),np.array(final_ddGs))

   