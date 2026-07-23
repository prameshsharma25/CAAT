import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ── Load your data ────────────────────────────────────────────────────────────
df = pd.read_csv('all_rocklin_mutated_updated_TM.csv')
ddg_all = df["ddG"]
structural_change_all=df["delta TM"]
attention_all = df["Normalized attention"]
confidence_all =df["delta pLDDT"]


# ── Filter to strongly destabilizing mutations (match Panel C) ────────────────
mask = ddg_all <= -3
ddg               = ddg_all[mask]
attention         = attention_all[mask]
structural_change = structural_change_all[mask]
confidence        = confidence_all[mask]

# ── Bin by attention decile ───────────────────────────────────────────────────
decile_labels = pd.qcut(pd.Series(attention).rank(method="first"), 
                        q=10, labels=False)
records = []
records2 = []
for dec in range(10):
    vals = structural_change[decile_labels == dec]
    vals2 = confidence[decile_labels == dec]
    median2 = np.median(vals2)
    median = np.median(vals)
    ci_low, ci_high = stats.bootstrap((vals,), np.median, confidence_level=0.95,
                                       random_state=42).confidence_interval
    ci_low2, ci_high2 = stats.bootstrap((vals2,), np.median, confidence_level=0.95,
                                       random_state=42).confidence_interval
    ci95 = (ci_high - ci_low) / 2
    ci952 = (ci_high2 - ci_low2) / 2
    records.append(dict(decile=dec + 1, mean=median, ci95=ci95, n=len(vals)))
    records2.append(dict(decile=dec + 1, mean=median2, ci95=ci952, n=len(vals2)))

res = pd.DataFrame(records)
res2 = pd.DataFrame(records2)

# ── Plot ──────────────────────────────────────────────────────────────────────
# Set font family to sans-serif and prioritize Helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']
fig, ax = plt.subplots(figsize=(10, 3))

GRAY   = "#AAAAAA"
BLUE   = "#2166AC"   # match Panel C palette if needed
RED    = "#D6604D"
BLACK = "#000000"

# Scatter (background) — jittered decile integer on x so points are evenly spread
jitter = np.random.uniform(-0.35, 0.35, size=len(attention))
ax.scatter(decile_labels + 1 + jitter, structural_change, s=3, alpha=0.07,
           color=GRAY, rasterized=True, zorder=1, label="_nolegend_")

# Decile means + 95% CI
ax.errorbar(res["decile"], res["mean"], yerr=res["ci95"],
            fmt="o", color=BLACK, ecolor=BLACK, elinewidth=1.5,
            capsize=4, capthick=1.5, markersize=8, zorder=3,
            label="Decile median ± 95% CI")

# Flat reference line at 0
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)

# ── Annotate correlation ───────────────────────────────────────────────────────
r, p = stats.pearsonr(attention, structural_change)
ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
        fontsize=11, color="black")
n_total = len(structural_change)
ax.text(0.05, 0.85, f"n = {n_total:,}", transform=ax.transAxes,
        fontsize=11, color="black")

# ── Labels & style ────────────────────────────────────────────────────────────
ax.set_xlabel("Attention decile", fontsize=16)
ax.set_ylabel("Change in predicted\nstructure (ΔTM)", fontsize=16)
ax.set_xticks(range(1, 11))
ax.set_xlim(0.4, 10.7)


ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.tick_params(labelsize=14)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, frameon=False, loc="upper left",
          bbox_to_anchor=(0.18, 0.98))

plt.tight_layout()
plt.savefig("fig1E.pdf", dpi=300, bbox_inches="tight")

plt.clf()


