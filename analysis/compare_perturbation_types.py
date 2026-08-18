"""
Compare per-layer stripped CKA across all perturbation types (char, token,
shuffle, typo, word) on a single model.

Reads `results/{model}/{ptb_type}/{pct}/evals.csv` (the layout written by
`main.py`) and produces:
  - a per-layer CKA line plot at a chosen percentage,
  - a type x layer CKA heatmap at that percentage,
  - a mean-CKA-vs-percentage line plot across all types.

Usage: python compare_perturbation_types.py [--model gpt2] [--pct 25]
"""
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import MODEL_INFO


PTB_TYPES = ['char', 'token', 'shuffle', 'typo', 'word', 'synonym']
PTB_NAMES = ['Char substitution', 'Token substitution', 'Token shuffling',
             'Typo (QWERTY)', 'Word substitution', 'Synonym substitution']
COLORS = ['#2166ac', '#d6604d', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']


def get_cka_prefix(df):
    """Auto-detect CKA column prefix (activation_cka_ or cka_)."""
    for c in df.columns:
        if c.startswith('activation_cka_layer_'):
            return 'activation_cka_'
        if c.startswith('cka_layer_'):
            return 'cka_'
    raise ValueError(f"No CKA columns found in evals.csv (columns: {list(df.columns)[:10]}...)")


def load_type_data(model, ptb_type, pct):
    path = f'results/{model}/{ptb_type}/{pct}/evals.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=lambda c: c.startswith('activation_cka_') or c.startswith('cka_'))
    prefix = get_cka_prefix(df)
    cols = [c for c in df.columns if c.startswith(prefix + 'layer_')]
    layers = sorted({int(c.replace(prefix + 'layer_', '')) for c in cols})
    return {L: df[f'{prefix}layer_{L}'].mean() for L in layers}


def load_per_sample_cka(model, ptb_type, pct):
    """Per-sample mean CKA (over all layers) for significance testing."""
    path = f'results/{model}/{ptb_type}/{pct}/evals.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=lambda c: c.startswith('activation_cka_') or c.startswith('cka_'))
    prefix = get_cka_prefix(df)
    cols = sorted([c for c in df.columns if c.startswith(prefix + 'layer_')],
                  key=lambda c: int(c.replace(prefix + 'layer_', '')))
    return df[cols].mean(axis=1)


def run_stats(model):
    """Mann-Whitney tests separating the regimes + early-layer CKA slope."""
    from scipy import stats

    print('\n' + '=' * 70)
    print('STATISTICAL SEPARATION (per-sample mean CKA, Mann-Whitney U, two-sided)')
    print('=' * 70)
    pairs = [('char', 'token'), ('char', 'word'), ('typo', 'word'),
             ('token', 'shuffle'), ('word', 'synonym')]
    for pct in [5, 25]:
        samples = {ptb: load_per_sample_cka(model, ptb, pct) for ptb in PTB_TYPES}
        print(f'\n--- pct={pct}% ---')
        for a, b in pairs:
            if samples[a] is None or samples[b] is None:
                continue
            u, p = stats.mannwhitneyu(samples[a], samples[b], alternative='two-sided')
            flag = '***' if p < 0.001 else ('*' if p < 0.05 else '')
            print(f'  {a:>8} ({samples[a].mean():.3f}) vs {b:<8} '
                  f'({samples[b].mean():.3f}): U={u:>6.0f}  p={p:>8.2e} {flag}')

    print('\nEARLY-LAYER CKA SLOPE (layers 0-4, more negative = faster early collapse)')
    for ptb in PTB_TYPES:
        d = load_type_data(model, ptb, 25)
        if d is None:
            continue
        xs = [L for L in range(5) if L in d]
        ys = [d[L] for L in xs]
        if len(xs) >= 2:
            slope, _, _, _, _ = stats.linregress(xs, ys)
            print(f'  {ptb:>8}: slope={slope:+.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2', help="Model key in config.MODEL_INFO")
    parser.add_argument('--pct', type=int, default=25, help="Percentage for the per-layer/heatmap figures")
    parser.add_argument('--out-dir', default='figures')
    args = parser.parse_args()

    num_layers = MODEL_INFO[args.model]['num_layers']
    layers = list(range(num_layers + 1))
    os.makedirs(args.out_dir, exist_ok=True)

    pcts = [5, 10, 25, 50]

    # Load per-layer CKA means: ptb -> pct -> {layer: cka}
    data = {}
    for ptb in PTB_TYPES:
        data[ptb] = {}
        for pct in pcts:
            data[ptb][pct] = load_type_data(args.model, ptb, pct)

    # --- Print summary table (mean CKA over layers) ---
    print('=' * 70)
    print(f'CROSS-TYPE CKA SUMMARY - model={args.model} (1 = identical, 0 = unrelated)')
    print('=' * 70)
    header = f"{'Type':<20}" + ''.join(f'{p:>10}' for p in pcts)
    print(header)
    print('-' * len(header))
    for ptb in PTB_TYPES:
        row = f'{ptb:<20}'
        for pct in pcts:
            d = data[ptb][pct]
            if d is None:
                row += f'{"N/A":>10}'
            else:
                row += f'{np.mean(list(d.values())):>10.4f}'
        print(row)

    run_stats(args.model)

    # --- Figure A: per-layer CKA at --pct ---
    pct = args.pct
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f'Per-Layer Stripped CKA by Perturbation Type ({args.model}, {pct}%)', fontsize=13)
    for ptb, name, color in zip(PTB_TYPES, PTB_NAMES, COLORS):
        d = data[ptb][pct]
        if d is None:
            continue
        xs = sorted(d)
        ax.plot(xs, [d[L] for L in xs], 'o-', color=color, label=name, linewidth=2)
    ax.axhline(1.0, color='black', lw=0.5, alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA to clean')
    ax.set_ylim(0, 1.02)
    ax.set_xticks(layers)
    ax.legend()
    plt.tight_layout()
    out_a = os.path.join(args.out_dir, f'cross_type_cka_by_layer.png')
    plt.savefig(out_a, bbox_inches='tight', dpi=150)
    print(f'Saved {out_a}')

    # --- Figure B: type x layer heatmap at --pct ---
    matrix = np.full((len(PTB_TYPES), len(layers)), np.nan)
    for i, ptb in enumerate(PTB_TYPES):
        d = data[ptb][pct]
        if d is None:
            continue
        for L, v in d.items():
            matrix[i, L] = v
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(matrix, ax=ax, cmap='viridis', vmin=0, vmax=1,
                yticklabels=PTB_NAMES, xticklabels=[f'L{L}' for L in layers])
    ax.set_title(f'Per-Layer Stripped CKA Heatmap ({args.model}, {pct}%)')
    ax.set_xlabel('Layer')
    plt.tight_layout()
    out_b = os.path.join(args.out_dir, 'cross_type_cka_heatmap.png')
    plt.savefig(out_b, bbox_inches='tight', dpi=150)
    print(f'Saved {out_b}')

    # --- Figure C: mean CKA vs perturbation % ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for ptb, name, color in zip(PTB_TYPES, PTB_NAMES, COLORS):
        xs = [p for p in pcts if data[ptb][p] is not None]
        ys = [np.mean(list(data[ptb][p].values())) for p in xs]
        if ys:
            ax.plot(xs, ys, 'o-', color=color, label=name, linewidth=2)
    ax.set_xlabel('Perturbation %')
    ax.set_ylabel('Mean stripped CKA (over layers)')
    ax.set_title(f'Mean CKA vs Perturbation Percentage ({args.model})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_c = os.path.join(args.out_dir, 'cross_type_cka_vs_pct.png')
    plt.savefig(out_c, bbox_inches='tight', dpi=150)
    print(f'Saved {out_c}')


if __name__ == '__main__':
    main()
