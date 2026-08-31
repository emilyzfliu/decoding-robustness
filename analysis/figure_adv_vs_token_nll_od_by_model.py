"""
Figure: NLL and output divergence, Adversarial (hotflip) vs. Random token,
pct=30, all 6 models -- grouped bar chart, one bar pair per model.

Unlike CKA/intrinsic-dim (which are per-layer and plotted as line-over-depth
in figure_adv_vs_token_by_model.py), nll/nll_base/output_divergence are
single scalars per sample, so a barplot (as in figure_n_candidates.py) is
the right form here, not a layer-depth line.

Matches figure_generation.ipynb's conventions: drop_duplicates(subset='sample')
before aggregating, first N_SAMPLES samples only (some dirs hold up to 748
cached samples from earlier over-generation; the paper's reported numbers use
the first 300), 95% CI via seaborn's built-in bootstrap ci=95, same
Adversarial/Random-token color scheme as the rest of the adv-vs-token figures.

Usage: python analysis/figure_adv_vs_token_nll_od_by_model.py
"""
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE = "/Users/niyathiallu/Desktop/ad_peturb-results_30"
MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'qwen2.5_0.5b', 'qwen2.5_1.5b']
N_SAMPLES = 300
COLORS = {'Adversarial': 'tab:blue', 'Random token': 'tab:orange'}
LABEL_FOR = {'hotflip': 'Adversarial', 'token': 'Random token'}


def main():
    rows = []
    for model in MODELS:
        for ptb in ['hotflip', 'token']:
            path = f'{BASE}/{model}/{ptb}/30/evals.csv'
            df = pd.read_csv(path, usecols=['sample', 'nll', 'nll_base', 'output_divergence'])
            df = df.drop_duplicates(subset='sample')
            df = df[df['sample'] < N_SAMPLES]
            for _, r in df.iterrows():
                rows.append({
                    'model': model,
                    'condition': LABEL_FOR[ptb],
                    'nll': r['nll'],
                    'delta_nll': r['nll'] - r['nll_base'],
                    'output_divergence': r['output_divergence'],
                })
    data = pd.DataFrame(rows)

    os.makedirs('figs_all/adv', exist_ok=True)

    specs = [
        ('nll', 'NLL', 'Sequence NLL: Adversarial vs. Random Token, by model (pct=30)', 'nll_by_model.png'),
        ('delta_nll', 'ΔNLL (perturbed − clean)',
         'ΔNLL: Adversarial vs. Random Token, by model (pct=30)', 'delta_nll_by_model.png'),
        ('output_divergence', 'Output divergence',
         'Output Divergence: Adversarial vs. Random Token, by model (pct=30)', 'output_divergence_by_model.png'),
    ]
    for col, ylabel, title, fname in specs:
        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
        sns.barplot(
            data=data, x='model', y=col, hue='condition', order=MODELS,
            hue_order=['Adversarial', 'Random token'], palette=COLORS,
            errorbar=('ci', 95), ax=ax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=20)
        for container in ax.containers:
            if hasattr(container, 'datavalues'):
                ax.bar_label(container, fmt='%.2f', padding=3, fontsize=8, rotation=90)
        ax.margins(y=0.18)
        ax.legend(title='', loc='upper left')
        fig.tight_layout()
        out_path = f'figs_all/adv/{fname}'
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'wrote {out_path}')

    print('\nSummary (mean ± std, n):')
    for col, ylabel, *_ in specs:
        print(f'\n-- {ylabel} --')
        summary = data.groupby(['model', 'condition'])[col].agg(['mean', 'std', 'count'])
        print(summary.reindex(MODELS, level='model').to_string())


if __name__ == '__main__':
    main()
