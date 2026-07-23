'''
Figure 3C using xcl1 data
'''

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Helvetica'

# Make figure 3C
def plot_sasa(df):
    fig, ax = plt.subplots()

    gating_residues = [40, 45]
    modulating_residues = [14]
    nongating_residues = [36, 38, 47]
    gating_labels = {40: 'B2.5', 45: 'b2b3.12'}
    modulating_labels = {14: 'cxb1.5'}
    nongating_labels = {36: 'B2.1', 38: 'B2.3', 47: 'B3.2'}

    # Label offsets
    gating_offsets = {
        40: (25, -13),     # B2.5 
        45: (25, -15),    # b2b3.12 
    }
    nongating_offsets = {
        36: (-20, -10),    # B2.1
        38: (27, -2),       # B2.3
        47: (22, -2),     # B3.2
    }
    modulating_offsets = {
        14: (25, -8)       # cxb1.5 
    }

    mask_gating = df['Residue number'].isin(gating_residues)
    mask_modulating = df['Residue number'].isin(modulating_residues)
    mask_nongating = df['Residue number'].isin(nongating_residues)
    mask_highlight = mask_gating | mask_nongating | mask_modulating

    x_all = df['Normalized attention score']
    y_all = 1 - df['Normalized relative SASA']

    r, p = pearsonr(x_all, y_all)

    # Trendline
    coef = np.polyfit(x_all, y_all, 1)
    x_line = np.linspace(x_all.min(), x_all.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    ax.plot(x_line, y_line, color='black', linewidth=2, zorder=1)

    # Background points
    ax.scatter(x_all[~mask_highlight], y_all[~mask_highlight],
               c='darkgray', s=40, alpha=0.7, zorder=2, label='All residues')

    # Non-gating residues (black)
    ax.scatter(x_all[mask_nongating], y_all[mask_nongating],
               c='black', s=80, zorder=4, label='Non-gating residues')

    # Gating residues (yellow)
    ax.scatter(x_all[mask_gating], y_all[mask_gating],
               c='#ca9823', s=80, zorder=4, label='Gating residues')

    # Modulating residues (light yellow)
    ax.scatter(x_all[mask_modulating], y_all[mask_modulating],
           c='#E2C883', s=80, zorder=4, label='Modulating residues')

    # Annotations — gating
    # for _, row in df.loc[mask_gating].iterrows():
    #     res = int(row['Residue number'])
    #     dx, dy = gating_offsets[res]
    #     ax.annotate(
    #         gating_labels[res],
    #         xy=(row['Normalized attention score'], 1 - row['Normalized relative SASA']),
    #         xytext=(dx, dy), textcoords='offset points',
    #         fontsize=16, color='#ca9823', ha='center', va='center', zorder=5
    #     )

    # # Annotations — modulating
    # for _, row in df.loc[mask_modulating].iterrows():
    #     res = int(row['Residue number'])
    #     dx, dy = modulating_offsets[res]
    #     ax.annotate(
    #         modulating_labels[res],
    #         xy=(row['Normalized attention score'], 1 - row['Normalized relative SASA']),
    #         xytext=(dx, dy), textcoords='offset points',
    #         fontsize=16, color='#E2C883', ha='center', va='center', zorder=5
    #     )

    # # Annotations — non-gating
    # for _, row in df.loc[mask_nongating].iterrows():
    #     res = int(row['Residue number'])
    #     dx, dy = nongating_offsets[res]
    #     ax.annotate(
    #         nongating_labels[res],
    #         xy=(row['Normalized attention score'], 1 - row['Normalized relative SASA']),
    #         xytext=(dx, dy), textcoords='offset points',
    #         fontsize=16, color='black', ha='center', va='center', zorder=5
    #     )

    # r value — bottom right
    ax.text(0.95, 0.05, f'r={r:.2f}',
        transform=ax.transAxes,
        fontsize=16,
        ha='right',
        verticalalignment='bottom')

    # Legend — upper left
    ax.legend(frameon=False, fontsize=12, loc='upper left', bbox_to_anchor=(-0.05, 1.03))

    ax.tick_params(axis='both', labelsize=16, length=7, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.set_xlabel('Normalized attention', fontsize=16)
    ax.set_ylabel('Relative burial (1-SASA)', fontsize=16)

    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    plt.savefig('./fig3C.pdf', dpi=300, bbox_inches='tight')


# Helper to calculate Pearson's coefficient
def get_pearson(df):
    x = df['Normalized attention score']
    y_c = df['Normalized contacts']
    y_s = df['Normalized relative SASA']

    contacts_r, contacts_p = pearsonr(x, y_c)
    sasa_r, sasa_p = pearsonr(x, y_s)

    print(f'contacts r & p-value: ', contacts_r, contacts_p)
    print(f'SASA r & p-value: ', sasa_r, sasa_p)


def main():
    path = f'./xcl1_ss_data.csv'

    df = pd.read_csv(path)
    t_df = df.loc[df['Amino acid'] == 'T']
    v_df = df.loc[df['Amino acid'] == 'V']

    min_t = t_df['Relative SASA'].min()
    max_t = t_df['Relative SASA'].max()
    min_v = v_df['Relative SASA'].min()
    max_v = v_df['Relative SASA'].max()

    for index, row in df.iterrows():
        if row['Amino acid'] == 'T':
            df.at[index, 'Normalized relative SASA'] = (row['Relative SASA'] - min_t) / (max_t - min_t)
        elif row['Amino acid'] == 'V':
            df.at[index, 'Normalized relative SASA'] = (row['Relative SASA'] - min_v) / (max_v - min_v)
        else:
            df.at[index, 'Normalized relative SASA'] = row['Relative SASA']

    plot_sasa(df)


if __name__ == "__main__":
    main()