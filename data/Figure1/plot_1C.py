'''
Code to plot Figure 1C:
Number of destabilizing mutations per attention decile / all mutations
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']


# Helper function for number of destabilizing mutations
def compute_summary(df, mask):
    df['is_destab'] = mask

    summary = df.groupby('attention_bin')['is_destab'].agg(
        N_total='count',
        N_destab='sum'
    ).reset_index()

    summary['fraction'] = summary['N_destab'] / summary['N_total']
    summary['stderr'] = np.sqrt(
        summary['fraction'] * (1 - summary['fraction']) / summary['N_total']
    )
    return summary


# Figure 1C
def plot_num_destabilizing():
    df_natural = pd.read_csv('./rocklin_mutated_natural.csv')
    df_design  = pd.read_csv('./rocklin_mutated_design.csv')
    df = pd.concat([df_natural, df_design], ignore_index=True)

    # Rank-based decile binning
    decile_labels = pd.qcut(
        df['Normalized attention'].rank(method='first'),
        q=10, labels=False
    )  # 0–9
    df['attention_bin'] = decile_labels + 1  # 1–10

    x = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(10, 3))

    s1 = compute_summary(df, df['ddG'] <= -3)

    ax.plot(x, s1['fraction'], linestyle='--', color='0.6', linewidth=1.2)
    ax.errorbar(
        x, s1['fraction'], yerr=s1['stderr'],
        fmt='none', ecolor='black', elinewidth=1.0, capsize=2
    )
    ax.scatter(x, s1['fraction'], s=60, color='black', zorder=3,
               label='Destabilizing (ΔΔG ≤ −3)')

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(range(1, 11), rotation=0)
    ax.set_xlim(0.4, 10.7)
    ax.set_xlabel('Attention decile', fontsize=16)
    ax.set_ylabel('Fraction destabilizing\nmutations (ΔΔG ≤ −3)', fontsize=16)
    ax.set_title('')
    ax.tick_params(labelsize=14)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    plt.savefig('./fig1C.pdf', dpi=300, bbox_inches='tight')


def main():
    plot_num_destabilizing()


if __name__ == "__main__":
    main()