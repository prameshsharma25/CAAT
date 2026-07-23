import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import sys


def read_info():
    # initializing dictionary to hold information for scatter plot
    keys_list = ['x_vals', 'y_vals', 'fold']
    info_dict = dict.fromkeys(keys_list)

    x_vals = []
    y_vals = []
    folds = []

    df = pd.read_csv('variants.csv')
    # iterating through each row of dataframe
    variant = ''
    for index, row in df.iterrows():
        # checking if the variant column is not nan
        if not pd.isna(row['Variant']):
            variant = row['Variant']
        if row['pLDDT'] >= 72:
            plddt = row['pLDDT']
            if row['TM-score vs chemokine'] > 0.5 and row['TM-score vs chemokine'] > row['TM-score vs dimer']:
                fold = 'chemokine'
            elif row['TM-score vs dimer'] > 0.5 and row['TM-score vs dimer'] > row['TM-score vs chemokine']:
                fold = 'dimer'
            else:
                fold = 'neither'
        # plddt = row['pLDDT']
        # if row['TM-score vs chemokine'] > 0.5 and row['TM-score vs chemokine'] > row['TM-score vs dimer']:
        #     fold = 'chemokine'
        # elif row['TM-score vs dimer'] > 0.5 and row['TM-score vs dimer'] > row['TM-score vs chemokine']:
        #     fold = 'dimer'
        # else:
        #     fold = 'neither'
        x_vals.append(variant)
        y_vals.append(plddt)
        folds.append(fold)

    info_dict['x_vals'] = x_vals
    info_dict['y_vals'] = y_vals
    info_dict['folds'] = folds

    return info_dict
    

def make_scatter(info_dict):
    color_map = {
        'chemokine': '#AE0639',
        'dimer': '#CA9823',
        'neither': 'black'
    }

    x = info_dict['x_vals']   
    y = info_dict['y_vals']    
    folds = info_dict['folds']
    
    colors = [color_map[f] for f in folds]

    fig, ax = plt.subplots(figsize=(6, 10))

    ax.scatter(y, x, c=colors, s=100, edgecolor='black')
    ax.invert_yaxis()

    # left y axis anc colors
    anc_colors = ['#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639',
                  '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#AE0639', '#CA9823', '#AE0639', '#CA9823', 
                  '#CA9823', '#CA9823']

    n = len(anc_colors)
    tick_locs = np.arange(n)
    ax.set_yticks(tick_locs)
    ax.set_ylim(n - 0.5, -0.5)   


    for tick_label, color in zip(ax.get_yticklabels(), anc_colors):
        tick_label.set_color(color=color)
        tick_label.set_path_effects([pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

    right_triplets = [
        'YTR', 'YTR', 'LTR', 'YTR', 'YTR', 'YIR', 'YTL', 'YTR', 'YTR',
        'LLL', 'LLL', 'LLL', 'LLL', 'LTI', 'LII', 'ILT', 'LII', 'LIL', 'LIL',
    ]

    right_colors_triplets = [
        ('#AE0639',  '#AE0639',  '#AE0639'), 
        ('#AE0639',  '#AE0639',  '#AE0639'),
        ('#CA9823', '#AE0639',  '#AE0639'),   
        ('#AE0639',  '#AE0639',  '#AE0639'),
        ('#AE0639',  '#AE0639',  '#AE0639'),
        ('#AE0639',  '#CA9823', '#AE0639'),   
        ('#AE0639',  '#AE0639',  '#CA9823'), 
        ('#AE0639',  '#AE0639',  '#AE0639'),
        ('#AE0639',  '#AE0639',  '#AE0639'),
        ('#CA9823', '#CA9823', '#CA9823'), 
        ('#CA9823', '#CA9823', '#CA9823'),
        ('#CA9823', '#CA9823', '#CA9823'),
        ('#CA9823', '#CA9823', '#CA9823'),
        ('#CA9823', '#AE0639', '#CA9823'),
        ('#CA9823', '#CA9823', '#CA9823'),  
        ('#CA9823', '#CA9823', 'black'),
        ('#CA9823', '#CA9823', '#CA9823'),
        ('#CA9823', '#CA9823', '#CA9823'),  
        ('#CA9823', '#CA9823', '#CA9823'),
    ]

    ax2 = ax.twinx()
    ax2.set_yticks(tick_locs)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels([''] * n)
    ax2.tick_params(axis='y', labelright=True, labelleft=False)

    # spacing for characters
    x0 = 1.02
    dx = 0.1  # spacing between right y axis labels

    # draw multicolor characters
    for y_pos, triplet, color_triplet in zip(tick_locs, right_triplets, right_colors_triplets):
        for i, (ch, col) in enumerate(zip(triplet, color_triplet)):
            t = ax2.text(
                x0 + i * dx,
                y_pos,
                ch,
                transform=ax2.get_yaxis_transform(),
                va='center',
                ha='left',
                fontsize=15,
                color=col,
            )
            t.set_path_effects([pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])

    number_labels = ['14', '43', '48']
    top_y = 1.0
    x0_2 = 1
    for i, num in enumerate(number_labels):
        ax2.text(
            x0_2 + i * dx,
            top_y,
            num,
            transform=ax2.transAxes,
            ha='left',
            va='bottom',
            fontsize=15
        )
    
    ax2.text(
        x0_2 + 0.04 + dx,     
        top_y + 0.03,   
        'Position',
        transform=ax2.transAxes,
        ha='center',
        va='bottom',
        fontsize=15
    )

    ax.set_xlabel('plDDT', fontname='Helvetica', fontsize=20)
    ax.set_ylabel('Variants', fontname='Helvetica', fontsize=20)
    ax.tick_params(axis='both', which='major',
                   labelfontfamily='Helvetica', labelsize=15, length=8, width=2)
    ax.tick_params(axis='x', which='major', labelsize=20)

    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    plt.savefig('./S3_af2.pdf', dpi=600)
    plt.show()


def main():
    info_dict = read_info()
    make_scatter(info_dict)

if __name__ == "__main__":
    main()