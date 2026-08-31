"""
Figure: Adversarial (hotflip) vs. Random token, pct=30, all 6 models -- matches
the exact layout of the user's own figure_generation.ipynb small-multiples
convention (one subplot per condition, all models overlaid with markers,
blue shades = GPT family by size, orange shades = Qwen family by size,
shared y-axis across subplots), just with 2 conditions (Adversarial, Random
token) instead of the notebook's original 6 (char/typo/token/shuffle/word/
synonym).

This is the middle ground between figure_adv_vs_token_by_model.py (one
subplot per model, only 2 lines each, easy to read but 6 panels) and
figure_adv_vs_token_single.py (all 12 lines on one axes, most compact but
busiest): 2 panels, 6 lines each, models distinguished by the notebook's
usual size-ordered shade scheme since 6 lines-per-panel is exactly the
regime that scheme was designed for.

Matches figure_generation.ipynb's conventions: drop_duplicates(subset='sample'),
first N_SAMPLES samples only, 95% CI via 1.96*std/sqrt(n), marker='o'.

Usage: python analysis/figure_adv_vs_token_by_condition.py
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
N_SAMPLES = 300  # some dirs hold up to 748 cached samples; paper's reported numbers use the first 300
MODEL_SERIES = {
    'gpt': ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    'qwen': ['qwen2.5_0.5b', 'qwen2.5_1.5b'],
}
SERIES_BASE_COLOR = {'gpt': 'tab:blue', 'qwen': 'tab:orange'}
METRICS = ['activation_cka', 'intrinsic_dim_change', 'intrinsic_dim_mknn_change']
YLABELS = {'activation_cka': 'CKA', 'intrinsic_dim_change': 'Δ Intrinsic Dim (2NN)',
           'intrinsic_dim_mknn_change': 'Δ Intrinsic Dim (KNN-MLE)'}
LABEL_FOR = {'hotflip': 'Adversarial', 'token': 'Random token'}
CONDITIONS = ['hotflip', 'token']


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
            for ptb in CONDITIONS:
                path = f'{BASE}/{m}/{ptb}/30/evals.csv'
                df = pd.read_csv(path).drop_duplicates(subset='sample')
                dfs[(m, ptb)] = df[df['sample'] < N_SAMPLES]

    model_color = {}
    for series_name, models in MODEL_SERIES.items():
        shades = size_ordered_shades(SERIES_BASE_COLOR[series_name], len(models))
        for model, color in zip(models, shades):
            model_color[model] = color

    os.makedirs('figs_all/adv', exist_ok=True)
    for metric in METRICS:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, ptb in zip(axs, CONDITIONS):
            for series_name, models in MODEL_SERIES.items():
                for model in models:
                    color = model_color[model]
                    num_layers = MODEL_INFO[model]['num_layers']
                    layers = list(range(1, num_layers + 1))
                    layers_plt = [l / num_layers for l in layers]
                    mean, std, n = mean_std_n_across_layers(dfs[(model, ptb)], layers, metric)
                    ci = 1.96 * (std / np.sqrt(n))
                    ax.plot(layers_plt, mean, marker='o', markersize=4, color=color,
                             label=model, linewidth=1.6)
                    ax.fill_between(layers_plt, mean - ci, mean + ci, alpha=0.12, color=color)
            ax.set_title(LABEL_FOR[ptb])
            ax.set_xlabel('Layer Depth')
            ax.legend(fontsize=8, loc='best')

        axs[0].set_ylabel(YLABELS[metric])
        fig.suptitle(metric.upper(), y=1.02)
        fig.tight_layout()
        out_path = f'figs_all/adv/{metric}_by_condition.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
