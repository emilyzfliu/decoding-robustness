"""
Figure: Adversarial (hotflip) vs. Random token, gpt2, one figure per metric,
3 subplots per figure (pct=5, 30, 50) -- extends the notebook's Figure 7
(which only covered pct=30) using the "fixed model, vary pct" layout from
Figures 2/3.

Matches figure_generation.ipynb's conventions: drop_duplicates(subset='sample')
before aggregating, 95% CI via 1.96*std/sqrt(n), same color scheme as Figure 7
(colors = {'Adversarial': 'tab:blue', 'Random token': 'tab:orange'}).

Usage: python analysis/figure_adv_vs_token_by_pct.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATHS = {
    (5, 'hotflip'):  '/Users/niyathiallu/Desktop/ad_peturb_results_5,50/gpt2/hotflip/5/evals.csv',
    (5, 'token'):    '/Users/niyathiallu/Desktop/ad_peturb_results_5,50/gpt2/token/5/evals.csv',
    (30, 'hotflip'): '/Users/niyathiallu/Desktop/ad_peturb-results_30/gpt2/hotflip/30/evals.csv',
    (30, 'token'):   '/Users/niyathiallu/Desktop/ad_peturb-results_30/gpt2/token/30/evals.csv',
    (50, 'hotflip'): '/Users/niyathiallu/Desktop/ad_peturb_results_5,50/gpt2/hotflip/50/evals.csv',
    (50, 'token'):   '/Users/niyathiallu/Desktop/ad_peturb_results_5,50/gpt2/token/50/evals.csv',
}
PCTS = [5, 30, 50]
N_LAYERS = 12  # gpt2's transformer blocks (layer 0 = embeddings, excluded to match paper's stated convention)
LAYERS = list(range(1, N_LAYERS + 1))
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
    for (pct, ptb), path in PATHS.items():
        df = pd.read_csv(path).drop_duplicates(subset='sample')
        dfs[(pct, ptb)] = df

    os.makedirs('figs_all/adv', exist_ok=True)
    for metric in METRICS:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
        for ax, pct in zip(axs, PCTS):
            for ptb in ['hotflip', 'token']:
                label = LABEL_FOR[ptb]
                df = dfs[(pct, ptb)]
                mean, std, n = mean_std_n_across_layers(df, LAYERS, metric)
                ci = 1.96 * (std / np.sqrt(n))
                ax.plot(LAYERS, mean, marker='o', label=label, color=COLORS[label])
                ax.fill_between(LAYERS, mean - ci, mean + ci, alpha=0.15, color=COLORS[label])
            ax.set_title(f'pct = {pct}')
            ax.set_xlabel('Layer')
        axs[0].set_ylabel(YLABELS[metric])
        handles, labels_ = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(f'{YLABELS[metric]}: Adversarial vs. Random Token (model=gpt2)', y=1.16)
        fig.tight_layout()
        out_path = f'figs_all/adv/{metric}_by_pct.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
