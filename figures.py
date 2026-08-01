import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib.colors as mcolors

import Levenshtein
from datasets import load_dataset
import random
from transformers import AutoTokenizer
import os
import argparse
from parse import parse
from matplotlib.lines import Line2D


from ablations import entropy_slope_by_head

parser = argparse.ArgumentParser(description="Generate figures from base experiment results.")
parser.add_argument('--model', default='gpt2', help="Model key in config.MODEL_INFO (default: gpt2)")
parser.add_argument('--out-dir', default='figures', help="Where to write figure/CSV outputs")
args = parser.parse_args()
MODEL = args.model
OUT_DIR = args.out_dir

# Write figure/CSV outputs to a local figures/ dir (inputs are read from results/).
os.makedirs(OUT_DIR, exist_ok=True)

ptb_types = ['token', 'char', 'shuffle']
ptb_names = ['Token Substitution', 'Char substitution', 'Token Shuffling']
percentages = [[x*5 for x in range(1, 11)] if x != 'shuffle' else [x*5 for x in range(1, 21)] for x in ptb_types]


# Set up a baseline
baseline = pd.read_csv(f'results/{MODEL}/char/0/evals.csv')

##### Figure 1 #####
print('Figure 1')

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Behavioral Metrics', fontsize=14)

def get_log_ppl_stats(df, col="nll"):
    # `nll` is already in log-space (per-token cross-entropy), so no log() here.
    nll = df[col].to_numpy()
    nll = nll[np.isfinite(nll)]

    return {
        "center": np.median(nll),      # robust sequence-level NLL
        "low": np.percentile(nll, 25),
        "high": np.percentile(nll, 75),
    }

for i, PTB_TYPE in enumerate(ptb_types):
    base_stats = get_log_ppl_stats(baseline)

    means = [base_stats["center"]]
    lows = [base_stats["low"]]
    highs = [base_stats["high"]]

    for j in range(1, 11):
        pct = 5 * j
        seq = pd.read_csv(f"results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv")

        stats = get_log_ppl_stats(seq)
        means.append(stats["center"])
        lows.append(stats["low"])
        highs.append(stats["high"])

    xs = np.array([x * 5 for x in range(11)])
    axs[0].plot(xs, means, label=ptb_names[i])
    axs[0].fill_between(xs, lows, highs, alpha=0.3)

axs[0].set_title("Sequence-level NLL / log perplexity")
axs[0].set_xlabel("Perturbation %")
axs[0].set_ylabel("log perplexity")
axs[0].legend()

for i, PTB_TYPE in enumerate(ptb_types):

    means = [np.median(baseline['output_divergence'])]
    lows  = [np.percentile(baseline['output_divergence'], 25)]
    highs = [np.percentile(baseline['output_divergence'], 75)]
    for j in range(1, 11):
        pct = 5*j
        seq = pd.read_csv(f'results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv')
        means.append(np.median(seq['output_divergence']))
        lows.append(np.percentile(seq['output_divergence'], 25))
        highs.append(np.percentile(seq['output_divergence'], 75))

    xs = np.array([x*5 for x in range(11)])
    axs[1].plot(xs, means, label=ptb_names[i])
    axs[1].fill_between(xs, lows, highs, alpha=0.3)

axs[1].set_title('Output Divergence')
axs[1].set_xlabel('Perturbation %')
axs[1].legend()


for i, PTB_TYPE in enumerate(ptb_types):

    means = [np.median(baseline['logit_kl'])]
    lows  = [np.percentile(baseline['logit_kl'], 25)]
    highs = [np.percentile(baseline['logit_kl'], 75)]
    for j in range(1, 11):
        pct = 5*j
        seq = pd.read_csv(f'results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv')
        means.append(np.median(seq['logit_kl']))
        lows.append(np.percentile(seq['logit_kl'], 25))
        highs.append(np.percentile(seq['logit_kl'], 75))

    xs = np.array([x*5 for x in range(11)])
    axs[2].plot(xs, means, label=ptb_names[i])
    axs[2].fill_between(xs, lows, highs, alpha=0.3)
axs[2].set_title('Logit KL Divergence')
axs[2].set_xlabel('Perturbation %')

axs[2].legend()

plt.savefig(f'{OUT_DIR}/behavioral.png', bbox_inches='tight', dpi=150)

####################

##### Figure 2 #####
print('Figure 2')

from src.perturbs import perturb

def divergence(list_a, list_b):
    [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(list_a, list_b)]

def generate_input_div(ptb_type, ptb_pct):
    SEQ_LEN = 128 
    rng = random.Random(1)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]

    rng_data = random.Random(1)

    texts = rng_data.sample(texts, 100)

    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer)

    dists = []

    for x, y in zip(texts, texts_perturbed):
        dists.append(Levenshtein.distance(x, y)/max(len(x), len(y)))
    
    dists = np.array(dists)
    
    return np.median(dists), np.percentile(dists, 25), np.percentile(dists, 75)

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Input vs Output Divergence', fontsize=14)

for idx in range(3):
    ptb = ptb_types[idx]

    inputs_mean = [0]
    inputs_low = [0]
    inputs_high = [0]
    for pct in range(5,55,5):
        med, lo, hi = generate_input_div(ptb, pct)
        inputs_mean.append(med)
        inputs_low.append(lo)
        inputs_high.append(hi)
    
    xs = np.array([x*5 for x in range(11)])
    axs[idx].plot(xs, inputs_mean, label='Input Divergence')
    axs[idx].fill_between(xs, inputs_low, inputs_high, alpha=0.3)


    means = [np.median(baseline['output_divergence'])]
    lows  = [np.percentile(baseline['output_divergence'], 25)]
    highs = [np.percentile(baseline['output_divergence'], 75)]
    for j in range(1, 11):
        pct = 5*j
        seq = pd.read_csv(f'results/{MODEL}/{ptb}/{pct}/evals.csv')
        means.append(np.median(seq['output_divergence']))
        lows.append(np.percentile(seq['output_divergence'], 25))
        highs.append(np.percentile(seq['output_divergence'], 75))

    xs = np.array([x*5 for x in range(11)])
    axs[idx].plot(xs, means, label="Output Divergence")
    axs[idx].fill_between(xs, lows, highs, alpha=0.3)

    axs[idx].set_title(ptb_names[idx])
    axs[idx].set_xlabel('Perturbation %')
    axs[idx].legend()

plt.savefig(f'{OUT_DIR}/in_vs_out.png', bbox_inches='tight', dpi=150)

#####################

###### Figure 3 #####
print('Figure 3')

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Per-Layer Activation Cosine Similarity', fontsize=14)


for i, PTB_TYPE in enumerate(ptb_types):

    mean_matrix = []

    for pct in percentages[i]:
        seq =  pd.read_csv(f'results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv')
        means = []
        for j in range(12):
            means.append(np.mean(seq[f'activation_cos_sim_layer_{j}']))
        
        mean_matrix.append(means)

    mean_matrix = np.array(mean_matrix)

    sns.heatmap(mean_matrix, ax=axs[i], cmap='viridis', yticklabels=percentages[i])
    axs[i].set_title(ptb_names[i])
    axs[i].set_xlabel('Layer')
    axs[i].set_ylabel('Percentage')
plt.savefig(f'{OUT_DIR}/mean_activation_similarity.png')

####################

##### Figure 4 #####
print('Figure 4')

fig, axs = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Change in Entropy with Increasing Perturbation', fontsize=14)


for i, PTB_TYPE in enumerate(ptb_types):
    ax = axs[i]

    slopes = entropy_slope_by_head(PTB_TYPE, percentages[i])

    idx = np.argpartition(np.abs(slopes).ravel(), -5)[-5:]
    coords = np.unravel_index(idx, slopes.shape)

    print(PTB_TYPE, coords)
    vmax = np.abs(slopes).max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sns.heatmap(
        slopes,
        ax=ax,
        cmap='RdBu_r',
        norm=norm,
        # annot=True,
        fmt='.3f',
        xticklabels=[f'L{i}' for i in range(12)],
        yticklabels=[f'H{h}' for h in range(12)],
        # center=0
    )

    ax.set_title(ptb_names[i])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/entropy_slope_heatmap.png', bbox_inches='tight', dpi=150)

####################

##### Figure 5 #####
print('Figure 5')

def parse_info(dir_name):
    template ="l={layer}_h={head}_pct={ptb_pct}_{classif}"
    res =  parse(template, dir_name).named
    return {k: int(res[k]) if type(res[k]) == int else res[k] for k in res}

ablation_dfs = {}

for ptb in ptb_types:
    layers = []
    heads = []
    classification = []
    gap_nll = []
    gap_kl = []
    gap_output = []
    ablated_dir = f'results/{MODEL}/ablated/{ptb}'
    if not os.path.isdir(ablated_dir):
        print(f'  [skip] no ablation data at {ablated_dir}')
        continue
    for candidate in os.listdir(ablated_dir):
        metadata = parse_info(candidate)
        pct = metadata['ptb_pct']
        ablated = pd.read_csv(f'{ablated_dir}/{candidate}/evals.csv')
        orig =  pd.read_csv(f'results/{MODEL}/{ptb}/{pct}/evals.csv')
        orig_baseline = pd.read_csv(f'results/{MODEL}/char/0/evals.csv')

        # NLL (columns are already log-space: nll, nll_base)
        ablated_ptb_nll = np.mean(ablated['nll'])
        ablated_base_nll = np.mean(ablated['nll_base'])
        ablated_gap = ablated_ptb_nll - ablated_base_nll

        orig_ptb_nll = np.mean(orig['nll'])
        orig_base_nll = np.mean(orig_baseline['nll'])
        orig_gap = orig_ptb_nll - orig_base_nll

        gap_nll.append((orig_gap - ablated_gap) / orig_gap * 100)

        # KL
        ablated_kl = np.mean(ablated['logit_kl'])
        orig_kl = np.mean(orig['logit_kl'])

        gap_kl.append((orig_kl - ablated_kl) / orig_kl * 100)

        # Output div
        ablated_output = np.mean(ablated['output_divergence'])
        orig_output = np.mean(orig['output_divergence'])

        gap_output.append((orig_output - ablated_output) / orig_output * 100)

        layers.append(metadata['layer'])
        heads.append(metadata['head'])
        classif = 'diffuse' if metadata['classif'] == 'sink' else 'sink'
        classification.append(classif)

    ablation_dfs[ptb] = pd.DataFrame(
        {
            'layers': layers,
            'heads': heads,
            'classifications': classification,
            'nll': gap_nll,
            'kl': gap_kl,
            'output': gap_output
        }
    )

if ablation_dfs:
    ablation_dfs['token'].to_csv(f'{OUT_DIR}/token_ablation_ranks.csv')
    ablation_dfs['char'].to_csv(f'{OUT_DIR}/char_ablation_ranks.csv')
    ablation_dfs['shuffle'].to_csv(f'{OUT_DIR}/shuffle_ablation_ranks.csv')

CLASS_COLORS = {'propagator': '#2166ac', 'compensator': '#d6604d', 'neutral': '#888888'}

ENTROPY_MARKERS = {'diffuse': 'o', 'sink': 'D'} 

def classify(output_gap, threshold):
    if output_gap > threshold:
        return 'propagator'
    elif output_gap < -threshold:
        return 'compensator'
    return 'neutral'

lbl = 'output'

def plot_ablation_figure(ptb_types, ablation_dfs, lbl):
    all_gaps = [v for ptb in ptb_types for v in ablation_dfs[ptb][lbl]]
    ymin, ymax = min(all_gaps) - 0.05, max(all_gaps) + 0.05

    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle('Ablation Effects', fontsize=14)

    for idx, ptb in enumerate(ptb_types):
        df = ablation_dfs[ptb].copy()
        threshold = 0.15

        df['true_class'] = df['classifications']
        df = df.sort_values(lbl).reset_index(drop=True)

        heads = [(df.iloc[i]['layers'], df.iloc[i]['heads']) for i in range(len(df))]
        gap_reduction = list(df[lbl])
        colors = [CLASS_COLORS[classify(g, threshold)] for g in gap_reduction]
        markers = [ENTROPY_MARKERS.get(df.iloc[i]['true_class'], 'o') for i in range(len(df))]

        ax = axs[idx]
        xs = range(len(df))

        ax.axhline(0, color='black', linewidth=0.6, zorder=1)
        ax.axhline(threshold, color=CLASS_COLORS['propagator'], linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axhline(-threshold, color=CLASS_COLORS['compensator'], linewidth=0.8, linestyle='--', alpha=0.7)

        for xi, (yi, color, marker) in enumerate(zip(gap_reduction, colors, markers)):
            ax.scatter(xi, yi, c=color, marker=marker, s=60, zorder=3)

        ax.set_xticks(xs)
        ax.set_xticklabels([f'L{x}H{y}' for x, y in heads], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_title(ptb_names[idx], fontsize=11)
        if idx == 0:
            ax.set_ylabel(f'{lbl.capitalize()} gap reduction (%)', fontsize=9)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['propagator'],  markersize=8, label='Propagator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['compensator'], markersize=8, label='Compensator'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CLASS_COLORS['neutral'],     markersize=8, label='Neutral'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', markersize=8, label='Diffuse head'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#555555', markersize=8, label='Sink head'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()

    plt.savefig(f'{OUT_DIR}/ablations.png', bbox_inches='tight', dpi=150)

if ablation_dfs:
    plot_ablation_figure(ptb_types, ablation_dfs, lbl)
else:
    print('[skip] Figure 5 (ablations): no ablation results found')

####################

##### Figure 6: corrected representation similarity #####
# Raw cosine is confounded by massive-activation outlier dims (fake final-layer
# recovery). The stripped metrics drop the top-5 highest-variance dims per layer.

pct_pick = {'token': 25, 'char': 25, 'shuffle': 50}
layers = list(range(13))

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Per-Layer Representation Similarity: raw cosine (confounded) vs stripped CKA', fontsize=14)

for i, PTB_TYPE in enumerate(ptb_types):
    pct = pct_pick[PTB_TYPE]
    seq = pd.read_csv(f'results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv')
    raw_cos   = [np.mean(seq[f'activation_cos_sim_layer_{L}'])      for L in layers]
    strip_cka = [np.mean(seq[f'activation_cka_layer_{L}'])          for L in layers]

    axs[i].plot(layers, raw_cos,   'o--', color='#999999', label='raw cosine (confounded)')
    axs[i].plot(layers, strip_cka, '^-',  color='#2166ac', label='stripped CKA')
    axs[i].axhline(1.0, color='black', lw=0.4, alpha=0.5)
    axs[i].set_title(f'{ptb_names[i]} @ {pct}%')
    axs[i].set_xlabel('Layer')
    axs[i].set_ylim(0, 1.02)
    if i == 0:
        axs[i].set_ylabel('Similarity to clean')
    axs[i].legend(fontsize=9)

plt.savefig(f'{OUT_DIR}/representation_similarity_corrected.png', bbox_inches='tight', dpi=150)

####################

##### Figure 7: stripped-CKA heatmap (corrected version of Figure 3) #####

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Per-Layer Stripped CKA (massive-activation dims removed)', fontsize=14)

for i, PTB_TYPE in enumerate(ptb_types):
    mean_matrix = []
    for pct in percentages[i]:
        seq = pd.read_csv(f'results/{MODEL}/{PTB_TYPE}/{pct}/evals.csv')
        mean_matrix.append([np.mean(seq[f'activation_cka_layer_{L}']) for L in range(13)])
    mean_matrix = np.array(mean_matrix)

    sns.heatmap(mean_matrix, ax=axs[i], cmap='viridis', vmin=0, vmax=1,
                yticklabels=percentages[i], xticklabels=[f'L{L}' for L in range(13)])
    axs[i].set_title(ptb_names[i])
    axs[i].set_xlabel('Layer')
    axs[i].set_ylabel('Perturbation %')

plt.savefig(f'{OUT_DIR}/stripped_cka_heatmap.png', bbox_inches='tight', dpi=150)

####################

##### Figure 8: BPE vs Word-Level Comparison #####

def load_bpe_word_data(metric_prefix, layer_agg='mean'):
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

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('BPE vs Word-Level Substitution: Representation Metrics', fontsize=14)

metrics_config = [
    ('cka', 'CKA (↑ = more similar)'),
    ('intrinsic_dim_change', 'TwoNN Intrinsic Dim Change\n(↑ = expansion)'),
    ('intrinsic_dim_mknn_change', 'MKNN Intrinsic Dim Change\n(↑ = expansion)'),
]

for idx, (metric_prefix, ylabel) in enumerate(metrics_config):
    data = load_bpe_word_data(metric_prefix)
    xs = [5, 10, 25, 50]
    
    axs[idx].plot(xs, data['token'], marker='o', linewidth=2, label='BPE Token Substitution', color='#2166ac')
    axs[idx].plot(xs, data['word'], marker='s', linewidth=2, label='Word-Level Substitution', color='#d6604d')
    
    axs[idx].set_xlabel('Perturbation %')
    axs[idx].set_ylabel(ylabel)
    axs[idx].set_title(ylabel.split('(')[0].strip())
    axs[idx].legend()
    axs[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/bpe_vs_word_comparison.png', bbox_inches='tight', dpi=150)
print("Saved results/bpe_vs_word_comparison.png")

####################
