import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    files = glob.glob('Pos_*_top3_5s_all_AAs_plddt.csv')

    all_ratios = [[],[],[],[],[],[]]
    labels = ['cxb1.5', 'B2.1', 'B2.3', 'B2.5', 'b2b3.12', 'B3.2']
    fidx_to_label = {
        '14': 'cxb1.5',
        '36': 'B2.1',
        '38': 'B2.3',
        '40': 'B2.5',
        '45': 'b2b3.12',
        '47': 'B3.2'
    }

    for f in files:

        df = pd.read_csv(f)

        fidx = f.split('_')[1]

        # Exclude rows where name contains 'anc2i' or 'anc2l'
        filtered = df[~df['name'].str.contains('anc2i|anc2l', case=False, na=False)]

        # Count Chemokine and Dimer per aa
        counts = (
            filtered.groupby(['aa', 'fold'])
            .size()
            .unstack(fill_value=0))   # columns: chemokine, dimer

        # mask based on the index of counts, not df
        is_letter = counts.index.to_series().str.fullmatch(r'[A-Za-z]')


        # --- combine all single-letter aa into one group ---
        letter_counts = counts[is_letter].sum()
        non_letter_counts = counts[~is_letter]

        # add combined row
        combined = pd.concat([
            non_letter_counts,
            pd.DataFrame([letter_counts], index=[fidx])
        ])

        # compute ratio
        combined['ratio'] = combined['Dimer'] / (combined['Chemokine'] + combined['Dimer'])
        combined['ratio'] = combined['ratio'].fillna(0)

        print(combined.keys())

        combined.index = combined.index.astype(int)
        combined_sorted = combined.sort_index()

        print(type(combined_sorted.index[-1]))

        print(f)
        #print(combined)

        out = combined.copy()

        # make a numeric sort key from the index
        out['_sortkey'] = pd.to_numeric(out.index, errors='coerce')

        # sort numeric rows first; non-numeric rows (like letters_only) go last
        out = (
            out.sort_values('_sortkey', kind='mergesort', na_position='last')
            .drop(columns='_sortkey')
        )

        
        ratio_list = out['ratio'].tolist()

        idx = labels.index(fidx_to_label[fidx])

        all_ratios[idx] = ratio_list

    # your 6 lists
    data = np.array(all_ratios)

    # define colors
    colors = ['#ffffff',"#f2e6bf",'#ca9823']

    # create colormap
    cmap = LinearSegmentedColormap.from_list('custom_map', colors)

    fig, ax = plt.subplots(figsize=(6.99, 4.80))

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")

    
    #plt.imshow(data,cmap=cmap)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=16, rotation=45)
    ax.set_yticklabels(labels, fontsize=16)

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if labels[tick] in ['B2.5', 'b2b3.12']:
            label.set_fontsize(18)
            label.set_fontweight("bold")

    for tick, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        if labels[tick] in ['B2.5', 'b2b3.12']:
            label.set_fontsize(18)
            label.set_fontweight("bold")

    plt.xticks(ticks=range(6), labels=labels)
    plt.yticks(ticks=range(6), labels=labels)
    
    #plt.title('Only a subset of residues can shift\npredicted structural state',fontsize=23, pad=16)
    #plt.xlabel('Position')
    #plt.ylabel('Position')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction dimer predictions", fontsize=16,labelpad=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    import matplotlib.patches as patches

    # Example indices for 40 and 45 in your array
    # adjust these to match your matrix indexing
    i40 = labels.index('B2.5')
    i45 = labels.index('b2b3.12')
    j40 = labels.index('B2.5')
    j45 = labels.index('b2b3.12')

    # Draw a rectangle around the 40/45 sub-block
    x0 = min(i40, i45) - 0.5
    y0 = min(i40, i45) - 0.5
    width = abs(j45 - j40) + 1
    height = abs(i45 - i40) + 1


    rect = patches.Rectangle(
        (x0, y0), width, height,
        linewidth=3.0, edgecolor="black", facecolor="none"
    )
    
    ax.add_patch(rect)

    # Coordinates of the two shifting cells
    cell_1 = (j45-0.1, i40+0.1)   # upper-right cell in the black box
    cell_2 = (j40+0.1, i45-0.1)   # lower-left cell in the black box

    # Text position: upper-left of the black box
    text_x = min(j40, j45) - 0.8
    text_y = min(i40, i45) - 0.0   # because smaller row index is visually higher here

    ax.text(
        text_x, text_y,
        "state-shifting\nresidues",
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom"
    )

    start_x = text_x + 0.05
    start_y = text_y

    # Arrow to the first shifting cell
    ax.annotate(
        "",
        xy=cell_1,
        xytext=(start_x, start_y - 0.2),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black")
    )

    # Arrow to the second shifting cell
    ax.annotate(
        "",
        xy=cell_2,
        xytext=(start_x, start_y + 0.2),
        arrowprops=dict(arrowstyle="->", lw=2.0, color="black")
    )


    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    plt.subplots_adjust(left=0.16, right=0.88, top=0.82, bottom=0.19)
    

    plt.savefig('fig3D.pdf',dpi=300, bbox_inches='tight')


    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print(f"Heatmap axes size: {bbox.width:.3f} x {bbox.height:.3f} inches")
