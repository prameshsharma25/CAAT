'''
Code for S8: Contacts vs Attention for single sequence Rocklin data
'''

import os
import glob
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def plot_data(df, y_axis, size):
    fig, ax = plt.subplots()


    x = df['Normalized attention score']
    y_c = df['Normalized contacts']
    y_s = df['Relative SASA']

    contacts_r, contacts_p = pearsonr(x, y_c)
    sasa_r, sasa_p = pearsonr(x, y_s)

    if y_axis == 'contacts':
        df.plot.scatter(x='Normalized attention score', y='Normalized contacts', c='red', label='Contacts', ax=ax)

        plt.title(f"Pearson's correlation coefficient: {contacts_r}")        

        plt.legend()
        plt.ylabel('Normalized contacts')
        plt.savefig(f'../contacts_backbone_vs_attention_{size}.png')
    else:
        df.plot.scatter(x='Normalized attention score', y='Relative SASA', c='blue', label='Relative SASA', ax=ax)
    
        plt.title(f"Pearson's correlation coefficient:  {sasa_r}")
        plt.legend()
        plt.ylabel('Relative SASA')
        plt.savefig(f'../sasa_vs_attention_{size}.png')


def main():
    size = '200-300'
    size_path = f'../../../minimal_sequence_representation/final_list/pdbs/{size}_residues/'
    p_size = os.listdir(size_path)
    p_size = [p.split('.')[0] for p in p_size]
    path = f'../attention_contacts_backbone_sasa.csv'
    df = pd.read_csv(path)
    df_cleaned = df.dropna(subset=['Normalized contacts'])
    df_filtered = df_cleaned[df_cleaned['Protein'].isin(p_size)]
    y_axis = 'contacts'
    plot_data(df_filtered, y_axis, size)
    




if __name__ == "__main__":
    main()