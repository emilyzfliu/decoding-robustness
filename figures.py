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
from parse import parse
from matplotlib.lines import Line2D


from ablations import entropy_slope_by_head

# Write figure/CSV outputs to a local figures/ dir (inputs are read from results/).
os.makedirs('figures', exist_ok=True)

ptb_types = ['token', 'char', 'shuffle']
ptb_names = ['Token Substitution', 'Char substitution', 'Token Shuffling']
percentages = [[x*5 for x in range(1, 11)] if x != 'shuffle' else [x*5 for x in range(1, 21)] for x in ptb_types]


# Set up a baseline
baseline = pd.read_csv('results/char/0/evals.csv')

##### Figure 1 #####

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Behavioral Metrics', fontsize=14)

def get_log_ppl_stats(df, col="perplexity"):
    ppl = df[col].to_numpy()
    ppl = ppl[np.isfinite(ppl) & (ppl > 0)]

    log_ppl = np.log(ppl)

    return {
        "center": np.median(log_ppl),      # robust sequence-level NLL
        "low": np.percentile(log_ppl, 25),
        "high": np.percentile(log_ppl, 75),
    }

for i, PTB_TYPE in enumerate(ptb_types):
    base_stats = get_log_ppl_stats(baseline)

    means = [base_stats["center"]]
    lows = [base_stats["low"]]
    highs = [base_stats["high"]]

    for j in range(1, 11):
        pct = 5 * j
        seq = pd.read_csv(f"results/{PTB_TYPE}/{pct}/evals.csv")

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
        seq = pd.read_csv(f'results/{PTB_TYPE}/{pct}/evals.csv')
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
        seq = pd.read_csv(f'results/{PTB_TYPE}/{pct}/evals.csv')
        means.append(np.median(seq['logit_kl']))
        lows.append(np.percentile(seq['logit_kl'], 25))
        highs.append(np.percentile(seq['logit_kl'], 75))

    xs = np.array([x*5 for x in range(11)])
    axs[2].plot(xs, means, label=ptb_names[i])
    axs[2].fill_between(xs, lows, highs, alpha=0.3)
axs[2].set_title('Logit KL Divergence')
axs[2].set_xlabel('Perturbation %')

axs[2].legend()

plt.savefig(f'figures/behavioral.png', bbox_inches='tight', dpi=150)

####################

##### Figure 2 #####

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

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
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
        seq = pd.read_csv(f'results/{ptb}/{pct}/evals.csv')
        means.append(np.median(seq['output_divergence']))
        lows.append(np.percentile(seq['output_divergence'], 25))
        highs.append(np.percentile(seq['output_divergence'], 75))

    xs = np.array([x*5 for x in range(11)])
    axs[idx].plot(xs, means, label="Output Divergence")
    axs[idx].fill_between(xs, lows, highs, alpha=0.3)

    axs[idx].set_title(ptb_names[idx])
    axs[idx].set_xlabel('Perturbation %')
    axs[idx].legend()

plt.savefig(f'figures/in_vs_out.png', bbox_inches='tight', dpi=150)

#####################

###### Figure 3 #####
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Per-Layer Activation Cosine Similarity', fontsize=14)


for i, PTB_TYPE in enumerate(ptb_types):

    mean_matrix = []

    for pct in percentages[i]:
        seq =  pd.read_csv(f'results/{PTB_TYPE}/{pct}/evals.csv')
        means = []
        for j in range(12):
            means.append(np.mean(seq[f'activation_cos_sim_layer_{j}']))
        
        mean_matrix.append(means)

    mean_matrix = np.array(mean_matrix)

    sns.heatmap(mean_matrix, ax=axs[i], cmap='viridis', yticklabels=percentages[i])
    axs[i].set_title(ptb_names[i])
    axs[i].set_xlabel('Layer')
    axs[i].set_ylabel('Percentage')
plt.savefig(f'figures/mean_activation_similarity.png')

####################

##### Figure 4 #####


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
plt.savefig(f'figures/entropy_slope_heatmap.png', bbox_inches='tight', dpi=150)

####################

##### Figure 5 #####

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
    for candidate in os.listdir(f'results/ablated/{ptb}'):
        metadata = parse_info(candidate)
        pct = metadata['ptb_pct']
        ablated = pd.read_csv(f'results/ablated/{ptb}/{candidate}/evals.csv')
        orig =  pd.read_csv(f'results/{ptb}/{pct}/evals.csv')
        orig_baseline = pd.read_csv('results/char/0/evals.csv')

        # NLL
        ablated_ptb_nll = np.mean(np.log(ablated['perplexity']))
        ablated_base_nll = np.mean(np.log(ablated['perplexity_baseline']))
        ablated_gap = ablated_ptb_nll - ablated_base_nll

        orig_ptb_nll = np.mean(np.log(orig['perplexity']))
        orig_base_nll = np.mean(np.log(orig_baseline['perplexity']))
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

ablation_dfs['token'].to_csv('figures/token_ablation_ranks.csv')
ablation_dfs['char'].to_csv('figures/char_ablation_ranks.csv')
ablation_dfs['shuffle'].to_csv('figures/shuffle_ablation_ranks.csv')

CLASS_COLORS = {'propagator': '#2166ac', 'compensator': '#d6604d', 'neutral': '#888888'}

ENTROPY_MARKERS = {'diffuse': 'o', 'sink': 'D'} 

def classify(output_gap, threshold):
    if output_gap > threshold:
        return 'propagator'
    elif output_gap < -threshold:
        return 'compensator'
    return 'neutral'

lbl = 'output'

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
    ax.axhline( threshold, color=CLASS_COLORS['propagator'],  linewidth=0.8, linestyle='--', alpha=0.7)
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

plt.savefig(f'figures/ablations.png', bbox_inches='tight', dpi=150)

####################

##### Figure 6: corrected representation similarity #####
# Raw cosine is confounded by massive-activation outlier dims (fake final-layer
# recovery). The stripped metrics drop the top-5 highest-variance dims per layer.

pct_pick = {'token': 25, 'char': 25, 'shuffle': 50}
layers = list(range(13))

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Per-Layer Representation Similarity: raw cosine (confounded) vs stripped metrics', fontsize=14)

for i, PTB_TYPE in enumerate(ptb_types):
    pct = pct_pick[PTB_TYPE]
    seq = pd.read_csv(f'results/{PTB_TYPE}/{pct}/evals.csv')
    raw_cos   = [np.mean(seq[f'activation_cos_sim_layer_{L}'])      for L in layers]
    strip_cos = [np.mean(seq[f'activation_cos_stripped_layer_{L}']) for L in layers]
    strip_cka = [np.mean(seq[f'activation_cka_layer_{L}'])          for L in layers]

    axs[i].plot(layers, raw_cos,   'o--', color='#999999', label='raw cosine (confounded)')
    axs[i].plot(layers, strip_cos, 's-',  color='#d6604d', label='stripped cosine')
    axs[i].plot(layers, strip_cka, '^-',  color='#2166ac', label='stripped CKA')
    axs[i].axhline(1.0, color='black', lw=0.4, alpha=0.5)
    axs[i].set_title(f'{ptb_names[i]} @ {pct}%')
    axs[i].set_xlabel('Layer')
    axs[i].set_ylim(0, 1.02)
    if i == 0:
        axs[i].set_ylabel('Similarity to clean')
    axs[i].legend(fontsize=9)

plt.savefig(f'figures/representation_similarity_corrected.png', bbox_inches='tight', dpi=150)

####################

##### Figure 7: stripped-CKA heatmap (corrected version of Figure 3) #####

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Per-Layer Stripped CKA (massive-activation dims removed)', fontsize=14)

for i, PTB_TYPE in enumerate(ptb_types):
    mean_matrix = []
    for pct in percentages[i]:
        seq = pd.read_csv(f'results/{PTB_TYPE}/{pct}/evals.csv')
        mean_matrix.append([np.mean(seq[f'activation_cka_layer_{L}']) for L in range(13)])
    mean_matrix = np.array(mean_matrix)

    sns.heatmap(mean_matrix, ax=axs[i], cmap='viridis', vmin=0, vmax=1,
                yticklabels=percentages[i], xticklabels=[f'L{L}' for L in range(13)])
    axs[i].set_title(ptb_names[i])
    axs[i].set_xlabel('Layer')
    axs[i].set_ylabel('Perturbation %')

plt.savefig(f'figures/stripped_cka_heatmap.png', bbox_inches='tight', dpi=150)

####################