"""
Figure: Adversarial (hotflip) vs. Random token, pct=30, one figure per metric,
one subplot per model (6 models, 2x3 grid) -- combines the notebook's Figure 7
(adversarial vs. token content) with Figure 5's convention for cross-model
comparison (layer depth normalized to [0,1], since models have different
depths: gpt2=12, gpt2-medium=24, gpt2-large=36, gpt2-xl=48, qwen0.5b=24,
qwen1.5b=28 layers).

Matches figure_generation.ipynb's conventions: drop_duplicates(subset='sample'),
95% CI via 1.96*std/sqrt(n), same Adversarial/Random-token color scheme as
Figure 7.

Usage: python analysis/figure_adv_vs_token_by_model.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import MODEL_INFO

BASE = "/Users/niyathiallu/Desktop/ad_peturb-results_30"
MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'qwen2.5_0.5b', 'qwen2.5_1.5b']
METRICS = ['activation_cka', 'intrinsic_dim_change', 'intrinsic_dim_mknn_change']
YLABELS = {'activation_cka': 'CKA', 'intrinsic_dim_change': 'Δ Intrinsic Dim (2NN)',
           'intrinsic_dim_mknn_change': 'Δ Intrinsic Dim (KNN-MLE)'}
COLORS = {'Adversarial': 'tab:blue', 'Random token': 'tab:orange'}
LABEL_FOR = {'hotflip': 'Adversarial', 'token': 'Random token'}


def mean_std_n_across_layers(df, layers, metric):
    mean, std, n = [], [], []
    for l in layers:
        col = df[f'{metric}_layer_{l}']
        mean.append(col.mean())
        std.append(col.std())
        n.append(len(col))
    return np.array(mean), np.array(std), np.array(n)


def main():
    dfs = {}
    for m in MODELS:
        for ptb in ['hotflip', 'token']:
            path = f'{BASE}/{m}/{ptb}/30/evals.csv'
            dfs[(m, ptb)] = pd.read_csv(path).drop_duplicates(subset='sample')

    os.makedirs('figs_all/adv', exist_ok=True)
    for metric in METRICS:
        fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
        for i, model in enumerate(MODELS):
            ax = axs[i // 3, i % 3]
            num_layers = MODEL_INFO[model]['num_layers']
            layers = list(range(1, num_layers + 1))
            layers_plt = [l / num_layers for l in layers]

            for ptb in ['hotflip', 'token']:
                label = LABEL_FOR[ptb]
                mean, std, n = mean_std_n_across_layers(dfs[(model, ptb)], layers, metric)
                ci = 1.96 * (std / np.sqrt(n))
                ax.plot(layers_plt, mean, marker='o', markersize=4, label=label, color=COLORS[label])
                ax.fill_between(layers_plt, mean - ci, mean + ci, alpha=0.15, color=COLORS[label])

            ax.set_title(model)
            ax.set_xlabel('Layer Depth')

        axs[0, 0].set_ylabel(YLABELS[metric])
        axs[1, 0].set_ylabel(YLABELS[metric])
        handles, labels_ = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.04))
        fig.suptitle(f'{YLABELS[metric]}: Adversarial vs. Random Token (pct=30, all models)', y=1.08)
        fig.tight_layout()
        out_path = f'figs_all/adv/{metric}_by_model.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
