'''
Code for S8: Contacts vs Attention for single sequence Rocklin data
'''

import os
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path


plt.rcParams['font.family'] = 'Helvetica'

def plot_contacts(df, filename, title):
    fig, ax = plt.subplots()
    x = df['Normalized Attention']
    y = df['Normalized Contacts']

    r, p = pearsonr(x, y)


    ax.scatter(x, y, s=2, alpha=0.3, color='darkgray')

    # Trendline on top
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    ax.plot(x_line, y_line, color='black', linewidth=1, linestyle='dashed')

    ax.text(0.05, 0.95, f'r = {r:.2f}',
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top')

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=11)

    ax.set_title(title, fontsize=20, pad=10)
    ax.set_ylabel('Normalized Contacts', fontsize=16)
    ax.set_xlabel('Normalized Attention', fontsize=16)



    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def plot_sasa(df, filename, title):
    fig, ax = plt.subplots()  # default figsize to match reference

    df['Buriedness'] = 1 - df['Normalized relative SASA']
    x = df['Normalized Attention']
    y = df['Buriedness']

    r, p = pearsonr(x, y)

    ax.scatter(x, y, s=2, alpha=0.3, color='darkgray')

    # Trendline on top
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    ax.plot(x_line, y_line, color='black', linewidth=1, linestyle='dashed')

    ax.text(0.05, 0.95, f'r = {r:.2f}',
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top')

    ax.tick_params(axis='both', labelsize=16, length=7, width=2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.set_title(title, fontsize=20, pad=10)
    ax.set_ylabel('Relative burial (1-SASA)', fontsize=16)
    ax.set_xlabel('Normalized Attention', fontsize=16)

    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    plt.savefig(filename, dpi=300, bbox_inches='tight')


# def load_contacts(path, subset='Normalized Contacts'):
#     """Read a CSV and drop rows with missing contact values."""
#     return pd.read_csv(path).dropna(subset=[subset])

def load_contacts(path, x='Normalized Attention', y='Normalized Contacts'):
    return pd.read_csv(path).dropna(subset=[x, y])

def prepare_sasa(df):
    """Add per-amino-acid min-max normalized SASA."""
    df = df.copy()
    df['Normalized relative SASA'] = df['Relative SASA']
    aa = list('EQGVPMTASDNLHKICFRYW')
    mask = df['Amino acid'].isin(aa)
    df.loc[mask, 'Normalized relative SASA'] = (
        df.loc[mask]
          .groupby('Amino acid')['Relative SASA']
          .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )
    return df

def main():
    cath_path = './cath_msa_attention_contacts_backbone_sasa.csv'
    rocklin_path = './rocklin_ss_attention_ddg_sasa_contacts_backbone.csv'

    # SASA plot
    plot_sasa(prepare_sasa(load_contacts(cath_path)), './S8_sasa_cath_msa.png', 'MSA')

    # Contact plots — same pipeline, different files
    contact_jobs = [
        (cath_path, './S8_contacts_cath_msa.png', 'MSA'),
        (rocklin_path, './S8_contacts_rocklin_ss.png', 'Single Sequence'),
    ]
    for path, out, title in contact_jobs:
        plot_contacts(load_contacts(path), out, title)



if __name__ == "__main__":
    main()