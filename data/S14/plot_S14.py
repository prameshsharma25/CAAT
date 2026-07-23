import os
import glob
import pandas as pd
import sys
import statistics
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'
### Code that uses all native contacts and all mutated contacts and found the mean/median, regardless if the mutation actually changed contacts
def plot_lollipop_all(df, value_type):
    ordered_df = df.sort_values(by=f'native {value_type}')

    plt.hlines(y=range(1, len(df.index)+1), xmin=ordered_df[f'mutated {value_type}'], xmax=ordered_df[f'native {value_type}'], color='grey', alpha=0.4, zorder=1)
    plt.scatter(ordered_df[f'mutated {value_type}'], range(1, len(df.index)+1), color='darkgray', alpha=1, label='mutated')
    plt.scatter(ordered_df[f'native {value_type}'], range(1, len(df.index)+1), color='#AE0639', alpha=1, label='native')
    plt.legend()

    plt.yticks(range(1, len(df.index)+1), ordered_df['protein'], size=12)
    plt.xticks(size=12)
    plt.xlabel(f'{value_type.capitalize()} number of contacts', size=12)
    plt.ylabel('Protein', size=12)

    plt.tight_layout() 
    plt.savefig(f'./S14.png', dpi=300)
    plt.close()



def main():
    ### Code that ignores whether or not the mutated contacts are actually different from native
    df = pd.read_csv('./contacts_for_lollipop.csv')
    plot_lollipop_all(df, 'median')





if __name__ == "__main__":
    main()
