"""
Figure: NLL vs. HotFlip's n_candidates (gradient-shortlist size), gpt2, pct=30, n=300.

Matches figure_generation.ipynb's conventions: drop_duplicates(subset='sample')
before aggregating (evals.csv has one row per token position, duplicated across
127 rows per sample), 95% CI via 1.96 * std/sqrt(n), seaborn barplot like Figure 6.

Usage: python analysis/figure_n_candidates.py
"""
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# n_candidates=50 is the project default -- its data lives in the plain '30' dir
# (from the earlier full-pool run), not a '30_ncNN' suffixed one.
PATHS = {
    1:   '/Users/niyathiallu/Desktop/hotflip_ncand/30_nc1/evals.csv',
    5:   '/Users/niyathiallu/Desktop/hotflip_ncand/30_nc5/evals.csv',
    10:  '/Users/niyathiallu/Desktop/hotflip_ncand/30_nc10/evals.csv',
    20:  '/Users/niyathiallu/Desktop/hotflip_ncand/30_nc20/evals.csv',
    50:  '/Users/niyathiallu/Desktop/ad_peturb-results_30/gpt2/hotflip/30/evals.csv',
    100: '/Users/niyathiallu/Desktop/hotflip_ncand/30_nc100/evals.csv',
}
N_SAMPLES = 300

rows = []
for nc, path in PATHS.items():
    df = pd.read_csv(path, usecols=['sample', 'nll'])
    df = df.drop_duplicates(subset='sample')
    df = df[df['sample'] < N_SAMPLES]
    for v in df['nll']:
        rows.append({'n_candidates': str(nc), 'nll': v})

data = pd.DataFrame(rows)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
sns.barplot(
    data=data,
    x='n_candidates',
    y='nll',
    order=['1', '5', '10', '20', '50', '100'],
    color='tab:blue',
    errorbar=('ci', 95),
    ax=ax,
)
ax.set_title('HotFlip: Sequence NLL vs. Candidate Pool Size (model=gpt2, pct=30)', fontsize=11)
ax.set_xlabel('n_candidates')
ax.set_ylabel('NLL')

for container in ax.containers:
    if hasattr(container, 'datavalues'):  # skip the error-bar container
        ax.bar_label(container, fmt='%.2f', padding=14, fontsize=10)

ax.margins(y=0.12)  # headroom so labels don't collide with the plot's top edge
fig.tight_layout()
os.makedirs('figs_all/adv', exist_ok=True)
out_path = 'figs_all/adv/n_candidates_nll.png'
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f'wrote {out_path}')

print('\nSummary:')
summary = data.groupby('n_candidates')['nll'].agg(['mean', 'std', 'count'])
summary = summary.reindex(['1', '5', '10', '20', '50', '100'])
print(summary.to_string())
