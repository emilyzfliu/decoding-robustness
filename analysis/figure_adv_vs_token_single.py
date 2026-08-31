"""
Figure: Adversarial (hotflip) vs. Random token, pct=30, ALL 6 models combined
into a single axes per metric (not a grid) -- 12 lines per figure.

Uses the notebook's MODEL_SERIES / size_ordered_shades scheme for model
color (GPT family = blue shades by size, Qwen family = orange shades by
size), plus linestyle to carry the Adversarial/Random-token distinction
(solid = Adversarial, dashed = Random token), since color alone can't
carry both dimensions on one axes. Layer depth normalized to [0,1] (models
range 12-48 layers). Same drop_duplicates + 95% CI conventions as the rest
of the notebook.

Usage: python analysis/figure_adv_vs_token_single.py
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from config import MODEL_INFO

BASE = "/Users/niyathiallu/Desktop/ad_peturb-results_30"
MODEL_SERIES = {
    'gpt': ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    'qwen': ['qwen2.5_0.5b', 'qwen2.5_1.5b'],
}
SERIES_BASE_COLOR = {'gpt': 'tab:blue', 'qwen': 'tab:orange'}
METRICS = ['activation_cka', 'intrinsic_dim_change', 'intrinsic_dim_mknn_change']
YLABELS = {'activation_cka': 'CKA', 'intrinsic_dim_change': 'Δ Intrinsic Dim (2NN)',
           'intrinsic_dim_mknn_change': 'Δ Intrinsic Dim (KNN-MLE)'}
LABEL_FOR = {'hotflip': 'Adversarial', 'token': 'Random token'}
LINESTYLE_FOR = {'hotflip': '-', 'token': '--'}


def size_ordered_shades(base_color, n):
    base = np.array(mcolors.to_rgb(base_color))
    light = 0.5 + 0.5 * base
    return np.linspace(light, base, n)


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
    for series_models in MODEL_SERIES.values():
        for m in series_models:
            for ptb in ['hotflip', 'token']:
                path = f'{BASE}/{m}/{ptb}/30/evals.csv'
                dfs[(m, ptb)] = pd.read_csv(path).drop_duplicates(subset='sample')

    os.makedirs('figs_all/adv', exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        for series_name, models in MODEL_SERIES.items():
            shades = size_ordered_shades(SERIES_BASE_COLOR[series_name], len(models))
            for model, color in zip(models, shades):
                num_layers = MODEL_INFO[model]['num_layers']
                layers = list(range(1, num_layers + 1))
                layers_plt = [l / num_layers for l in layers]
                for ptb in ['hotflip', 'token']:
                    mean, std, n = mean_std_n_across_layers(dfs[(model, ptb)], layers, metric)
                    ci = 1.96 * (std / np.sqrt(n))
                    label = f'{model} ({LABEL_FOR[ptb]})'
                    ax.plot(layers_plt, mean, linestyle=LINESTYLE_FOR[ptb], color=color,
                             label=label, linewidth=1.6)
                    ax.fill_between(layers_plt, mean - ci, mean + ci, alpha=0.10, color=color)

        ax.set_xlabel('Layer Depth')
        ax.set_ylabel(YLABELS[metric])
        ax.set_title(f'{YLABELS[metric]}: Adversarial (solid) vs. Random Token (dashed), all models, pct=30')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8, ncol=1)
        fig.tight_layout()
        out_path = f'figs_all/adv/{metric}_all_models_single.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
