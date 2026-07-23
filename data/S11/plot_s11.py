import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── 1. Header + format detection ─────────────────────────────────────────────

def read_header(filepath):
    header = {}
    with open(filepath) as fh:
        first = fh.readline().strip()
        if first.startswith('#'):
            for part in first[1:].split('\t'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    header[k.strip()] = v.strip()
    return header


def detect_format(filepath):
    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if re.search(r'pos_\d+_[^/]+/pos_\d+_[A-Z]\d+[A-Z]_unrelaxed', line):
                return 'fmt1'
            elif re.search(r'pos_\d+_[a-z]\d+[a-z]', line):
                return 'fmt2'
            elif re.search(r'pos_\d+_[A-Z]\d+[A-Z]_unrelaxed', line):
                return 'fmt3'
    return 'fmt1'


# ── 2. Colors ─────────────────────────────────────────────────────────────────

COLOR_RED  = '#AE0639'
COLOR_GOLD = '#CA9823'
COLOR_GRAY = '#DDDDDD'
COLOR_WT   = '#FFFFFF'
AA_ORDER   = list('ACDEFGHIKLMNPQRSTVWY')

def get_fold_colors(fold1_color):
    if fold1_color == 'gold':
        return COLOR_GOLD, COLOR_RED
    return COLOR_RED, COLOR_GOLD


# ── 3. Parse ──────────────────────────────────────────────────────────────────

PATTERNS = {
    'fmt1': re.compile(
        r'pos_(\d+)_[^/]+/pos_\d+_([A-Z])(\d+)([A-Z])_unrelaxed.*?'
        r'TM-score=\s*([\d.]+)'
        r'(?:.*?TM-score=\s*([\d.]+))?',
        re.DOTALL
    ),
    'fmt2': re.compile(
        r'pos_(\d+)_([a-z])(\d+)([a-z]).*?'
        r'TM-score=\s*([\d.]+)'
        r'(?:.*?TM-score=\s*([\d.]+))?',
        re.DOTALL
    ),
    'fmt3': re.compile(
        r'pos_(\d+)_([A-Z])(\d+)([A-Z])_unrelaxed.*?'
        r'TM-score=\s*([\d.]+)'
        r'(?:.*?TM-score=\s*([\d.]+))?',
        re.DOTALL
    ),
}

def parse_scores(filepath, mode='two', swap=False):
    fmt = detect_format(filepath)
    pat = PATTERNS[fmt]
    print(f"Detected format: {fmt}")

    if mode == 'two':
        data = defaultdict(lambda: defaultdict(lambda: {'fold1': [], 'fold2': []}))
    else:
        data = defaultdict(lambda: defaultdict(list))
    wt_map = {}

    with open(filepath) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = pat.search(line)
            if not m:
                continue
            pos    = int(m.group(1))
            wt_aa  = m.group(2).upper()
            sub_aa = m.group(4).upper()
            tm1    = float(m.group(5))
            tm2    = float(m.group(6)) if m.group(6) else None
            wt_map[pos] = wt_aa

            if mode == 'one':
                data[pos][sub_aa].append(tm1)
            else:
                if tm2 is None:
                    continue
                fold1 = tm2 if swap else tm1
                fold2 = tm1 if swap else tm2
                data[pos][sub_aa]['fold1'].append(fold1)
                data[pos][sub_aa]['fold2'].append(fold2)

    return data, wt_map


# ── 4. Cell color helpers ─────────────────────────────────────────────────────

def cell_color_two(fold1_scores, fold2_scores, threshold, C1, C2):
    if not fold1_scores or not fold2_scores:
        return COLOR_GRAY, np.nan
    med1 = np.median(fold1_scores)
    med2 = np.median(fold2_scores)
    if med1 >= threshold or med2 >= threshold:
        return (C1, med1) if med1 >= med2 else (C2, med2)
    return COLOR_GRAY, max(med1, med2)


def cell_color_one(scores, threshold, C1):
    if not scores:
        return COLOR_GRAY, np.nan
    med = np.median(scores)
    return (C1, med) if med >= threshold else (COLOR_GRAY, med)


# ── 5. Consensus ──────────────────────────────────────────────────────────────

def consensus_two(data, wt_map, threshold, C1, C2):
    positions = sorted(data.keys())
    results   = []
    for pos in positions:
        wt = wt_map.get(pos)
        n1 = n2 = n_gray = 0
        for aa, scores in data[pos].items():
            if aa == wt:
                continue
            c, _ = cell_color_two(scores['fold1'], scores['fold2'], threshold, C1, C2)
            if c == C1:        n1     += 1
            elif c == C2:      n2     += 1
            else:              n_gray += 1
        total     = n1 + n2 + n_gray
        pct_fold1 = 100 * n1 / total if total > 0 else 0
        pct_fold2 = 100 * n2 / total if total > 0 else 0
        if n1 >= n2 and n1 >= n_gray:   consensus = C1
        elif n2 >= n_gray:              consensus = C2
        else:                           consensus = COLOR_GRAY
        results.append((pos, consensus, pct_fold1, pct_fold2))
    return results


def consensus_one(data, threshold, C1):
    positions = sorted(data.keys())
    results   = []
    for pos in positions:
        n_folded = n_gray = 0
        for aa, scores in data[pos].items():
            c, _ = cell_color_one(scores, threshold, C1)
            if c == C1: n_folded += 1
            else:       n_gray   += 1
        total      = n_folded + n_gray
        pct_folded = 100 * n_folded / total if total > 0 else 0
        consensus  = C1 if n_folded >= n_gray else COLOR_GRAY
        results.append((pos, consensus, pct_folded))
    return results


# ── 6. Plot collapsed (two-score) ─────────────────────────────────────────────

def plot_collapsed_two(filepath, out_png=None, label_fold='fold1', swap=False,
                       name_fold1='Fold1', name_fold2='Fold2', threshold=0.5,
                       fold1_color='red'):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    C1, C2       = get_fold_colors(fold1_color)
    data, wt_map = parse_scores(filepath, mode='two', swap=swap)
    summary      = consensus_two(data, wt_map, threshold, C1, C2)
    label_name   = name_fold2 if label_fold == 'fold2' else name_fold1

    n_pos = len(summary)
    fig, ax = plt.subplots(figsize=(max(6, n_pos * 0.55), 7))

    for j, (pos, color, pct_fold1, pct_fold2) in enumerate(summary):
        alpha     = pct_fold1/100 if color == C1 else pct_fold2/100 if color == C2 else 1.0
        pct_label = pct_fold1 if label_fold == 'fold1' else pct_fold2
        rect = mpatches.FancyBboxPatch(
            (j - 0.45, -0.45), 0.90, 1.1,
            boxstyle='round,pad=0.05', linewidth=0.5,
            edgecolor='#AAAAAA', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
        ax.text(j, 0.75, f'{pct_label:.0f}%',
                ha='center', va='bottom', fontsize=28,
                color='#333333', fontweight='bold', rotation=90)

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(-0.55, 1.2)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(p) for p, *_ in summary], fontsize=28)
    ax.set_yticks([])
    ax.set_xlabel('Position', fontsize=30)
    ax.set_title(f'Consensus prediction per position (% {label_name} labeled)', fontsize=30)
    ax.tick_params(length=0)

    fig.legend(handles=[
        mpatches.Patch(color=C1, label=name_fold1),
        mpatches.Patch(color=C2, label=name_fold2),
        mpatches.Patch(color=COLOR_GRAY, label='Unfolded'),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=30, frameon=False)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=600, bbox_inches='tight')
        print(f'Saved to {out_png}')
    else:
        plt.show()


# ── 7. Plot collapsed (one-score) ─────────────────────────────────────────────

def plot_collapsed_one(filepath, out_png=None, name_folded='Folded', threshold=0.5,
                       fold1_color='red'):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']

    C1, _    = get_fold_colors(fold1_color)
    data, _  = parse_scores(filepath, mode='one')
    summary  = consensus_one(data, threshold, C1)

    n_pos = len(summary)
    fig, ax = plt.subplots(figsize=(max(6, n_pos * 0.55), 7))

    for j, (pos, color, pct) in enumerate(summary):
        alpha = pct / 100 if color == C1 else 1.0
        rect = mpatches.FancyBboxPatch(
            (j - 0.45, -0.45), 0.90, 1.1,
            boxstyle='round,pad=0.05', linewidth=0.5,
            edgecolor='#AAAAAA', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
        ax.text(j, 0.75, f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=28,
                color='#333333', fontweight='bold', rotation=90)

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(-0.55, 1.2)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(p) for p, *_ in summary], fontsize=28)
    ax.set_yticks([])
    ax.set_xlabel('Position', fontsize=30)
    ax.set_title(f'Consensus prediction per position (% {name_folded} labeled)', fontsize=30)
    ax.tick_params(length=0)

    fig.legend(handles=[
        mpatches.Patch(color=C1, label=f'{name_folded} (≥{threshold})'),
        mpatches.Patch(color=COLOR_GRAY, label='Misfolded (<50% of AAs)'),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=30, frameon=False)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=600, bbox_inches='tight')
        print(f'Saved to {out_png}')
    else:
        plt.show()


# ── 8. Hardcoded runs ─────────────────────────────────────────────────────────

plot_collapsed_two(
    filepath    = './KaiB_V83S_all_aas_folded_only.txt',
    out_png     = 'S11.png',
    label_fold  = 'fold2',       
    swap        = False,
    name_fold1  = 'Ground State',
    name_fold2  = 'Fold Switched',
    threshold   = 0.6,
    fold1_color = 'red',          # 'red' or 'gold'
)



