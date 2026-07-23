'''
Plots figure 3A:
Rocklin unmutated data showing relationship between SASA and attention
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

plt.rcParams['font.family'] = 'Helvetica'


def plot_sasa(df):
    fig, ax = plt.subplots()  # default figsize to match reference

    df['Buriedness'] = 1 - df['Normalized relative SASA']
    x = df['Normalized Attention']
    y = df['Buriedness']

    r, p = pearsonr(x, y)

    print(r)
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

    # ax.set_title("Attention reflects structural \n burial across diverse proteins", fontsize=23, pad=16)
    ax.set_ylabel('Relative burial (1-SASA)', fontsize=16)
    ax.set_xlabel('Normalized Attention', fontsize=16)

    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    plt.savefig('./fig3A.pdf', dpi=300, bbox_inches='tight')



def main():
    path = './all_rocklin_unmutated_info.csv'
    df = pd.read_csv(path)
    df_cleaned = df.dropna(subset=['Normalized Contacts'])

    df_cleaned['Normalized relative SASA'] = df_cleaned['Relative SASA']

    amino_acids = ['R', 'P', 'A', 'G', 'D', 'E', 'K', 'V', 'N', 'S', 'Q', 'H', 'M', 'L', 'I', 'T']
    mask = df_cleaned['Amino acid'].isin(amino_acids)

    df_cleaned.loc[mask, 'Normalized relative SASA'] = df_cleaned[mask].groupby('Amino acid')['Relative SASA'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    plot_sasa(df_cleaned)


if __name__ == "__main__":
    main()