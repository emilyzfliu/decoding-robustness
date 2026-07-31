"""
Generate BPE vs Word comparison figures from existing results.
Does not require base experiments (char/token/shuffle/typo).

Usage: python plot_comparison.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def get_cka_prefix(df):
    """Auto-detect CKA column prefix (activation_cka_ or cka_)."""
    for c in df.columns:
        if c.startswith('activation_cka_layer_'):
            return 'activation_cka_'
        if c.startswith('cka_layer_'):
            return 'cka_'
    return 'activation_cka_'


def load_bpe_word_data(metric_prefix, layer_agg='mean'):
    """Load BPE vs word comparison data for a given metric prefix."""
    percentages = [5, 10, 25, 50]
    results = {'token': [], 'word': []}
    
    for ptb_type in ['token', 'word']:
        for pct in percentages:
            try:
                df = pd.read_csv(f'results/compare_bpe_word/{ptb_type}/{pct}/evals.csv')
                metric_cols = [c for c in df.columns if c.startswith(metric_prefix)]
                if metric_cols:
                    if layer_agg == 'mean':
                        layer_values = [df[col].mean() for col in metric_cols]
                        results[ptb_type].append(np.mean(layer_values))
                    else:
                        vals = [df[col].mean() for col in metric_cols if any(f'_{l}' in col for l in layer_agg)]
                        results[ptb_type].append(np.mean(vals) if vals else 0)
                else:
                    results[ptb_type].append(0)
            except FileNotFoundError:
                results[ptb_type].append(0)
    
    return results


# Set style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

# Figure 1: CKA, TwoNN, MKNN across perturbation levels
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('BPE vs Word-Level Substitution: Representation Metrics', fontsize=14, fontweight='bold')

metrics_config = [
    ('activation_cka', 'CKA (1 = identical)'),
    ('intrinsic_dim_change', 'TwoNN Intrinsic Dim Change\n(positive = expansion)'),
    ('intrinsic_dim_mknn_change', 'MKNN Intrinsic Dim Change\n(positive = expansion)'),
]

for idx, (metric_prefix, ylabel) in enumerate(metrics_config):
    data = load_bpe_word_data(metric_prefix)
    xs = [5, 10, 25, 50]
    
    axs[idx].plot(xs, data['token'], marker='o', linewidth=2.5, 
                  label='BPE Token Substitution', color='#2166ac', markersize=8)
    axs[idx].plot(xs, data['word'], marker='s', linewidth=2.5, 
                  label='Word-Level Substitution', color='#d6604d', markersize=8)
    
    axs[idx].set_xlabel('Perturbation %')
    axs[idx].set_ylabel(ylabel)
    axs[idx].legend()
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_xlim(0, 55)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/bpe_vs_word_comparison.png', bbox_inches='tight', dpi=150)
print("Saved results/bpe_vs_word_comparison.png")

# Figure 2: Per-layer CKA at each perturbation level
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Per-Layer CKA: BPE vs Word Substitution', fontsize=14, fontweight='bold')

for idx, pct in enumerate([5, 10, 25, 50]):
    row, col = idx // 2, idx % 2
    ax = axs[row, col]
    
    df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
    df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
    cka_prefix = get_cka_prefix(df_bpe)
    
    cka_bpe = [df_bpe[f'{cka_prefix}layer_{i}'].mean() for i in range(12)]
    cka_word = [df_word[f'{cka_prefix}layer_{i}'].mean() for i in range(12)]
    
    x = np.arange(12)
    width = 0.35
    
    ax.bar(x - width/2, cka_bpe, width, label='BPE Token', color='#2166ac', alpha=0.85)
    ax.bar(x + width/2, cka_word, width, label='Word-Level', color='#d6604d', alpha=0.85)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA')
    ax.set_title(f'{pct}% Perturbation')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in range(12)])
    ax.set_ylim(0.6, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('results/bpe_vs_word_per_layer_cka.png', bbox_inches='tight', dpi=150)
print("Saved results/bpe_vs_word_per_layer_cka.png")

# Check if TwoNN/MKNN columns exist before plotting
df_sample = pd.read_csv('results/compare_bpe_word/token/5/evals.csv')
has_twonn = any('intrinsic_dim_change_' in c for c in df_sample.columns)
has_mknn = any('intrinsic_dim_mknn_change_' in c for c in df_sample.columns)

if has_twonn:
    # Figure 3: TwoNN intrinsic dim per layer
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Layer TwoNN Intrinsic Dimension Change', fontsize=14, fontweight='bold')
    for idx, pct in enumerate([5, 10, 25, 50]):
        row, col = idx // 2, idx % 2
        ax = axs[row, col]
        df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
        df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
        twonn_bpe = [df_bpe[f'intrinsic_dim_change_layer_{i}'].mean() for i in range(12)]
        twonn_word = [df_word[f'intrinsic_dim_change_layer_{i}'].mean() for i in range(12)]
        x = np.arange(12); width = 0.35
        ax.bar(x - width/2, twonn_bpe, width, label='BPE Token', color='#2166ac', alpha=0.85)
        ax.bar(x + width/2, twonn_word, width, label='Word-Level', color='#d6604d', alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer'); ax.set_ylabel('TwoNN Dim Change'); ax.set_title(f'{pct}%')
        ax.set_xticks(x); ax.set_xticklabels([f'L{i}' for i in range(12)])
        ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('results/bpe_vs_word_twonn_per_layer.png', bbox_inches='tight', dpi=150)
    print("Saved results/bpe_vs_word_twonn_per_layer.png")
else:
    print("Skipping TwoNN figures (data not available)")

if has_mknn:
    # Figure 4: MKNN intrinsic dim per layer
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Layer MKNN Intrinsic Dimension Change', fontsize=14, fontweight='bold')
    for idx, pct in enumerate([5, 10, 25, 50]):
        row, col = idx // 2, idx % 2
        ax = axs[row, col]
        df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
        df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
        mknn_bpe = [df_bpe[f'intrinsic_dim_mknn_change_layer_{i}'].mean() for i in range(12)]
        mknn_word = [df_word[f'intrinsic_dim_mknn_change_layer_{i}'].mean() for i in range(12)]
        x = np.arange(12); width = 0.35
        ax.bar(x - width/2, mknn_bpe, width, label='BPE Token', color='#2166ac', alpha=0.85)
        ax.bar(x + width/2, mknn_word, width, label='Word-Level', color='#d6604d', alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer'); ax.set_ylabel('MKNN Dim Change'); ax.set_title(f'{pct}%')
        ax.set_xticks(x); ax.set_xticklabels([f'L{i}' for i in range(12)])
        ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('results/bpe_vs_word_mknn_per_layer.png', bbox_inches='tight', dpi=150)
    print("Saved results/bpe_vs_word_mknn_per_layer.png")
else:
    print("Skipping MKNN figures (data not available)")

# Figure 5: Combined view - All metrics at 25% perturbation
pct = 25
df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
cka_prefix = get_cka_prefix(df_bpe)

n_panels = 1 + (1 if has_twonn else 0) + (1 if has_mknn else 0)
fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
if n_panels == 1:
    axs = [axs]
fig.suptitle('All Metrics at 25% Perturbation (Per Layer)', fontsize=14, fontweight='bold')

x = np.arange(12)
panel_idx = 0

# Panel 1: CKA
cka_bpe = [df_bpe[f'{cka_prefix}layer_{i}'].mean() for i in range(12)]
cka_word = [df_word[f'{cka_prefix}layer_{i}'].mean() for i in range(12)]
axs[panel_idx].plot(x, cka_bpe, 'o-', color='#2166ac', label='BPE Token', linewidth=2, markersize=6)
axs[panel_idx].plot(x, cka_word, 's-', color='#d6604d', label='Word-Level', linewidth=2, markersize=6)
axs[panel_idx].set_xlabel('Layer')
axs[panel_idx].set_ylabel('CKA (1 = identical)')
axs[panel_idx].set_title('CKA')
axs[panel_idx].set_xticks(x)
axs[panel_idx].set_xticklabels([f'L{i}' for i in range(12)])
axs[panel_idx].legend()
axs[panel_idx].grid(True, alpha=0.3)
panel_idx += 1

# Panel 2: TwoNN
if has_twonn:
    twonn_bpe = [df_bpe[f'intrinsic_dim_change_layer_{i}'].mean() for i in range(12)]
    twonn_word = [df_word[f'intrinsic_dim_change_layer_{i}'].mean() for i in range(12)]
    axs[panel_idx].plot(x, twonn_bpe, 'o-', color='#2166ac', label='BPE Token', linewidth=2, markersize=6)
    axs[panel_idx].plot(x, twonn_word, 's-', color='#d6604d', label='Word-Level', linewidth=2, markersize=6)
    axs[panel_idx].axhline(0, color='black', linewidth=0.5)
    axs[panel_idx].set_xlabel('Layer')
    axs[panel_idx].set_ylabel('TwoNN Dim Change')
    axs[panel_idx].set_title('TwoNN Intrinsic Dim Change')
    axs[panel_idx].set_xticks(x)
    axs[panel_idx].set_xticklabels([f'L{i}' for i in range(12)])
    axs[panel_idx].legend()
    axs[panel_idx].grid(True, alpha=0.3)
    panel_idx += 1

# Panel 3: MKNN
if has_mknn:
    mknn_bpe = [df_bpe[f'intrinsic_dim_mknn_change_layer_{i}'].mean() for i in range(12)]
    mknn_word = [df_word[f'intrinsic_dim_mknn_change_layer_{i}'].mean() for i in range(12)]
    axs[panel_idx].plot(x, mknn_bpe, 'o-', color='#2166ac', label='BPE Token', linewidth=2, markersize=6)
    axs[panel_idx].plot(x, mknn_word, 's-', color='#d6604d', label='Word-Level', linewidth=2, markersize=6)
    axs[panel_idx].axhline(0, color='black', linewidth=0.5)
    axs[panel_idx].set_xlabel('Layer')
    axs[panel_idx].set_ylabel('MKNN Dim Change')
    axs[panel_idx].set_title('MKNN Intrinsic Dim Change')
    axs[panel_idx].set_xticks(x)
    axs[panel_idx].set_xticklabels([f'L{i}' for i in range(12)])
    axs[panel_idx].legend()
    axs[panel_idx].grid(True, alpha=0.3)
    panel_idx += 1

plt.tight_layout()
plt.savefig('results/all_metrics_25pct.png', bbox_inches='tight', dpi=150)
print("Saved results/all_metrics_25pct.png")

print("\nAll figures saved to results/")