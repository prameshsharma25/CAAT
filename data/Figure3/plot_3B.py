'''
Plots figure 3B:
Plots the change in TM-score upon masking different columns (random, high-attention, and matched-conservation)
for the CATH dataset
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'Helvetica'


def plot_boxplot_from_csv():
    df4 = pd.read_csv('./0.4_dtm_summary.csv')
    df5 = pd.read_csv('./0.5_dtm_summary.csv')
    df6 = pd.read_csv('./0.6_dtm_summary.csv')

    data = [
        df4['High attention'], df4['Random'], df4['Matched random'],
        df5['High attention'], df5['Random'], df5['Matched random'],
        df6['High attention'], df6['Random'], df6['Matched random'],
    ]
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]

    fig, ax = plt.subplots()  # default figsize to match reference

    box = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True, showfliers=False)

    colors = ['#CA9823', 'darkgray', '#AE0639', '#CA9823', 'darkgray', '#AE0639', '#CA9823', 'darkgray', '#AE0639']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    for median in box['medians']:
        median.set_color('black')

    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(['40', '50', '60'], fontsize=16)
    ax.tick_params(axis='both', labelsize=16, length=7, width=2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    gold_patch = mpatches.Patch(color='#CA9823', label='High-\nattention')
    gray_patch = mpatches.Patch(color='darkgray', label='Random')
    red_patch = mpatches.Patch(color='#AE0639', label='Matched-\nconservation')
    ax.legend(handles=[gold_patch, gray_patch, red_patch], fontsize=12, frameon=False, loc='lower left', bbox_to_anchor=(0.05, 0))

    ax.set_xlabel('% of MSA columns masked', fontsize=16)
    ax.set_ylabel('Change in TM-score upon \n column masking (ΔTM)', fontsize=16)
    #ax.set_title('AlphaFold uses high attention \n MSA columns disproportionately', fontsize=23)

    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    plt.savefig('./fig3B.pdf', dpi=300, bbox_inches='tight')



def main():
    plot_boxplot_from_csv()


if __name__ == "__main__":
    main()