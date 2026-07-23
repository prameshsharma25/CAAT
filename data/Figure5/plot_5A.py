import os
import glob
import pandas as pd
import sys
import statistics
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'


def plot_lollipop(df, value_type):
    ordered_df = df.sort_values(by=f'native contacts')

    plt.hlines(y=range(1, len(df.index)+1), xmin=ordered_df[f'mutated {value_type}'], xmax=ordered_df[f'native contacts'], color='grey', alpha=0.4, zorder=1)
    plt.scatter(ordered_df[f'mutated {value_type}'], range(1, len(df.index)+1), color='darkgray', alpha=1, label='mutated')
    plt.scatter(ordered_df[f'native contacts'], range(1, len(df.index)+1), color='#AE0639', alpha=1, label='native')
    plt.legend()

    plt.yticks(range(1, len(df.index)+1), ordered_df['protein'], size=12)
    plt.xticks(size=12)
    plt.xlabel(f'{value_type.capitalize()} number of native contacts', size=12)
    plt.ylabel('Protein', size=12)

    plt.tight_layout() 
    plt.savefig(f'./fig5E.png', dpi=300)
    plt.close()


def main():
    df = pd.read_csv('./contacts_for_lollipop_only_change.csv')
    plot_lollipop(df, 'median')



if __name__ == "__main__":
    main()
