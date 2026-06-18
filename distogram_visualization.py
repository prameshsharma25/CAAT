import streamlit as st
import numpy as np
import jax
import matplotlib.pyplot as plt
import py3Dmol
from stmol import showmol
import os
from Bio import Align

# 1. PAGE SETUP
st.set_page_config(page_title="Protein Distograms", layout="wide")
st.title("Protein Structure & Distogram Dashboard")

# 2. SIDEBAR CONTROLS
st.sidebar.header("Parameters")
seed = st.sidebar.slider("seed", 0, 5, 0)
model = st.sidebar.slider("model", 1, 5, 1)
recycle = st.sidebar.slider("recycle", 0, 3, 0)
main_loop = st.sidebar.slider("main_loop", 1, 48, 1)

st.sidebar.markdown("---")
st.sidebar.header("Display Toggles")
show_heatmaps = st.sidebar.checkbox("Show Heatmaps", value=True)
show_structures = st.sidebar.checkbox("Show 3D Structures", value=True)
show_distributions = st.sidebar.checkbox("Show Probability Distributions", value=True)
show_sequences = st.sidebar.checkbox("Show Sequence Alignment", value=True)

# 3. HELPER FUNCTIONS
@st.cache_data
def load_pair(seed, model, recycle, main_loop):
    s = f'{seed:03d}'
    p1 = f'xcl1_distograms/seed_{s}_model_{model}_recycle_{recycle}_main_loop_{main_loop}_distogram.npz'
    p2 = f'anc0_distograms/seed_{s}_model_{model}_recycle_{recycle}_main_loop_{main_loop}_distogram.npz'
    
    with np.load(p1) as f1, np.load(p2) as f2:
        d1 = {'bin_edges': f1['bin_edges'], 'logits': f1['logits']}
        d2 = {'bin_edges': f2['bin_edges'], 'logits': f2['logits']}
        
    return d1, d2

def build_xs(bin_edges):
    xs_local = [(2 + bin_edges[0]) / 2]
    for k in range(0, len(bin_edges) - 1):
        xs_local.append((bin_edges[k] + bin_edges[k + 1]) / 2)
    xs_local.append((bin_edges[-1] + 22) / 2)
    return np.array(xs_local)

def get_structure_path(prefix, seed, model, recycle, main_loop):
    return f'{prefix}_structures/seed_{seed:03d}_model_{model}_recycle_{recycle}_main_loop_{main_loop}_structure.pdb'

def render_pdb(pdb_path):
    try:
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()
        view = py3Dmol.view(width=400, height=400)
        view.addModel(pdb_str, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        showmol(view, height=400, width=400)
    except FileNotFoundError:
        st.error(f"PDB file not found: {pdb_path}")

def get_sequence_from_pdb(pdb_path):
    """Extracts the amino acid sequence by parsing the CA atoms in the PDB file."""
    if not os.path.exists(pdb_path):
        return ""
        
    aa_map = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
              'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
              'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
              'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    seq = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resName = line[17:20].strip()
                seq.append(aa_map.get(resName, 'X'))
    return "".join(seq)

# 4. MAIN LOGIC & RENDERING
try:
    # Load Data
    d1, d2 = load_pair(seed, model, recycle, main_loop)
    xs1, xs2 = build_xs(d1['bin_edges']), build_xs(d2['bin_edges'])
    
    # Calculate Distances
    soft1, soft2 = jax.nn.softmax(d1['logits']), jax.nn.softmax(d2['logits'])
    idx1 = np.array(jax.numpy.argmax(soft1, axis=2))
    idx2 = np.array(jax.numpy.argmax(soft2, axis=2))
    dist1, dist2 = xs1[idx1], xs2[idx2]
    
    # Lock the color scale
    global_vmin = min(xs1.min(), xs2.min())
    global_vmax = max(xs1.max(), xs2.max())

    # Get structure paths
    xcl1_path = get_structure_path('xcl1', seed, model, recycle, main_loop)
    anc0_path = get_structure_path('anc0', seed, model, recycle, main_loop)

    # Establish main columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"xcl1: seed={seed}, model={model}, rec={recycle}, loop={main_loop}")
    with col2:
        st.subheader(f"anc0: seed={seed}, model={model}, rec={recycle}, loop={main_loop}")

    # --- HEATMAPS ---
    if show_heatmaps:
        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            im1 = ax1.imshow(dist1, cmap='viridis', interpolation='nearest', vmin=global_vmin, vmax=global_vmax)
            # 1-indexing: set tick labels from 1 to N
            ax1.set_xticks(np.arange(0, dist1.shape[1], 10))
            ax1.set_xticklabels(np.arange(1, dist1.shape[1] + 1, 10))
            ax1.set_yticks(np.arange(0, dist1.shape[0], 10))
            ax1.set_yticklabels(np.arange(1, dist1.shape[0] + 1, 10))
            ax1.set_xlabel('Residue j')
            ax1.set_ylabel('Residue i')
            fig1.colorbar(im1, ax=ax1, label='Distance (Å)')
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im2 = ax2.imshow(dist2, cmap='viridis', interpolation='nearest', vmin=global_vmin, vmax=global_vmax)
            # 1-indexing: set tick labels from 1 to N
            ax2.set_xticks(np.arange(0, dist2.shape[1], 10))
            ax2.set_xticklabels(np.arange(1, dist2.shape[1] + 1, 10))
            ax2.set_yticks(np.arange(0, dist2.shape[0], 10))
            ax2.set_yticklabels(np.arange(1, dist2.shape[0] + 1, 10))
            ax2.set_xlabel('Residue j')
            ax2.set_ylabel('Residue i')
            fig2.colorbar(im2, ax=ax2, label='Distance (Å)')
            st.pyplot(fig2)

    # --- 3D STRUCTURES ---
    if show_structures:
        with col1:
            if not show_heatmaps: st.write("**3D Structure**")
            render_pdb(xcl1_path)
        with col2:
            if not show_heatmaps: st.write("**3D Structure**")
            render_pdb(anc0_path)
            
    # --- PROBABILITY DISTRIBUTIONS ---
    if show_distributions:
        st.markdown("---")
        st.header("Pairwise Distance Probability Distribution")
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**xcl1 Pair Selector**")
            seq_len_1 = soft1.shape[0]
            c3a, c3b = st.columns(2)
            # Sliders now display 1 to seq_len
            i_1_idx = c3a.slider("Residue i", 1, seq_len_1, 1, key="i_xcl1") - 1
            j_1_idx = c3b.slider("Residue j", 1, seq_len_1, 1, key="j_xcl1") - 1
            
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(xs1, soft1[i_1_idx, j_1_idx, :], marker='.', color='#2c728e') 
            ax3.fill_between(xs1, soft1[i_1_idx, j_1_idx, :], alpha=0.3, color='#2c728e')
            ax3.set_xlabel("Distance (Å)")
            ax3.set_ylabel("Probability")
            ax3.set_ylim(0, 1.05) 
            st.pyplot(fig3)
            
        with col4:
            st.write("**anc0 Pair Selector**")
            seq_len_2 = soft2.shape[0]
            c4a, c4b = st.columns(2)
            # Sliders now display 1 to seq_len
            i_2_idx = c4a.slider("Residue i", 1, seq_len_2, 1, key="i_anc0") - 1
            j_2_idx = c4b.slider("Residue j", 1, seq_len_2, 1, key="j_anc0") - 1
            
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.plot(xs2, soft2[i_2_idx, j_2_idx, :], marker='.', color='#20a486')
            ax4.fill_between(xs2, soft2[i_2_idx, j_2_idx, :], alpha=0.3, color='#20a486')
            ax4.set_xlabel("Distance (Å)")
            ax4.set_ylabel("Probability")
            ax4.set_ylim(0, 1.05)
            st.pyplot(fig4)

    # --- SEQUENCE ALIGNMENT (Hard-coded) ---
    if show_sequences:
        st.markdown("---")
        st.header("Static Sequence Alignment")
        
        # Hard-coded sequences with gaps
        seq1 = "VGSEVSDKRTCVSLTTQRLPVSRIKTYTITE---GSLRAVIFITKRGLKVCADPQATWVRDVVRSMDRKSNT"
        seq2 = "-----ARKSCCLKYTKRPLPLKRIKSYTIQSNEACNIKAIIFTTKKGRKICANPNEKWVQKAMKHLDK---K"
        
        # Determine the length for the ruler
        total_len = len(seq1)
        
        # Create the ruler
        ruler_top = ""
        ruler_bot = ""
        for i in range(total_len):
            pos = i + 1
            if pos % 10 == 0:
                ruler_top += str(pos)[0]
                ruler_bot += str(pos)[-1]
            else:
                ruler_top += " "
                ruler_bot += "."
        
        # Format display with alignment markers
        alignment_text = f"{'xcl1':>6}  {seq1}\n"
        alignment_text += f"{'anc0':>6}  {seq2}\n"
        alignment_text += f"{'':>6}  {ruler_top}\n"
        alignment_text += f"{'':>6}  {ruler_bot}\n"
        
        st.code(alignment_text, language="text")

except Exception as e:
    st.error(f"Error loading files. Check if the seed/model/loop combination exists. Detail: {e}")