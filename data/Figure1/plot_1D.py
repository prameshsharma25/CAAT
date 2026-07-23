'''
Code for Figure 1D:
Plots Rocklin unmutated SASA ddG <= -3 vs attention deciles
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica']


def normalize_sasa():
    path = './all_rocklin_unmutated.csv'
    df = pd.read_csv(path)
    df_cleaned = df.dropna(subset=['Normalized Contacts'])

    df_cleaned['Normalized relative SASA'] = df_cleaned['Relative SASA']

    amino_acids = ['R', 'P', 'A', 'G', 'D', 'E', 'K', 'V', 'N', 'S', 'Q', 'H', 'M', 'L', 'I', 'T']
    mask = df_cleaned['Amino acid'].isin(amino_acids)

    df_cleaned.loc[mask, 'Normalized relative SASA'] = df_cleaned[mask].groupby('Amino acid')['Relative SASA'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return df_cleaned


def plot_sasa_vs_attention():
    df = normalize_sasa()
    filtered_df = df[df['ddG'] <= -3].copy()
    filtered_df['Buriedness'] = 1 - filtered_df['Relative SASA']

    attention  = filtered_df['Normalized Attention'].values
    buriedness = filtered_df['Buriedness'].values

    decile_labels = pd.qcut(
        pd.Series(attention).rank(method='first'),
        q=10, labels=False
    ).values  # 0–9

    corr_df = filtered_df[['Normalized Attention', 'Buriedness']].dropna()
    r, p = pearsonr(corr_df['Normalized Attention'], corr_df['Buriedness'])
    print(f"r = {r:.2f}, p = {p:.3e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))

    # Group data by decile for ax.boxplot()
    data_by_decile = [buriedness[decile_labels == dec] for dec in range(10)]

    bp = ax.boxplot(
        data_by_decile,
        positions=range(1, 11),
        showfliers=False,
        patch_artist=False,
        boxprops=dict(color='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        medianprops=dict(color='black', linewidth=2.0),
        widths=0.6,
    )

    # Annotate correlation
    ax.text(0.05, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=11, color="black")

    # Labels & style
    ax.set_xlabel('Attention decile', fontsize=16)
    ax.set_ylabel('Relative burial (1-SASA)', fontsize=14)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(range(1, 11), rotation=0)
    ax.margins(x=0)
    ax.set_xlim(0.4, 10.7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax.tick_params(labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.savefig('./fig1D.pdf', dpi=300, bbox_inches='tight')


def main():
    plot_sasa_vs_attention()


if __name__ == '__main__':
    main()