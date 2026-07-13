import pandas as pd
import numpy as np
from scipy import stats
from config import MODEL_INFO
import matplotlib.pyplot as plt

import seaborn as sns

MODEL_PLOT_CONFIG = {
    'gpt2': {'name':'GPT-2 Small (117M)','color': 'lightsteelblue'},
    'gpt2-medium': {'name':'GPT-2 Medium (345M)','color': 'cornflowerblue'},
    'gpt2-large': {'name':'GPT-2 Large (762M)','color': 'blue'},
    'gpt2-xl': {'name':'GPT-2 XL (1542M)','color': 'navy'},
    'qwen2.5_0.5b': {'name':'Qwen-2.5 0.5B', 'color': 'lightcoral'},
    'qwen2.5_1.5b': {'name':'Qwen-2.5 1.5B', 'color': 'red'},
    'qwen2.5_7b': {'name':'Qwen-2.5 7B', 'color': 'darkred'},
}

def compute_activation_similarity_metrics(model, seq):
    layer_indices = [x for x in range(MODEL_INFO[model]['num_layers'])]
    values = [seq[f'activation_cos_sim_layer_{i}'].mean() for i in layer_indices]

    slope, _, r, p, _ = stats.linregress(layer_indices, values)
    auc = np.trapezoid(values, layer_indices)
    
    return slope, p, r**2, auc, values[-1]

# all in one, some data loss
def compute_attention_entropy_metrics_comparison(ptb_type, n_bins=5):
    all_data = {'model': [], 'depth_bin': [], 'slope': []}

    for model in MODEL_INFO.keys():
        num_layers = MODEL_INFO[model]['num_layers']
        head_indices = list(range(MODEL_INFO[model]['num_heads']))
        ptb_pcts = [x*5 for x in range(11)] if ptb_type != 'shuffle' else [x*5 for x in range(21)]

        entropy_cols = [f'attn_layer{i}_head_{h}_entropy_norm' for i in range(num_layers) for h in head_indices]
        dfs_by_pct = {}
        for pct in ptb_pcts:
            try:
                seq = pd.read_csv(f'results/{model}/{ptb_type}/{pct}/evals.csv', usecols=entropy_cols)
            except (FileNotFoundError, ValueError):
                seq = pd.read_csv(f'results/{model}/char/0/evals.csv', usecols=entropy_cols)
            dfs_by_pct[pct] = seq

        ptb_arr = np.array(ptb_pcts, dtype=float)
        x_centered = ptb_arr - ptb_arr.mean()
        denom = (x_centered ** 2).sum()

        bin_edges = np.linspace(0, 1, n_bins + 1)

        for i in range(num_layers):
            depth_frac = i / (num_layers - 1) if num_layers > 1 else 0
            b = min(np.digitize(depth_frac, bin_edges) - 1, n_bins - 1)

            means = np.array([
                [dfs_by_pct[pct][f'attn_layer{i}_head_{h}_entropy_norm'].mean() for h in head_indices]
                for pct in ptb_pcts
            ])
            y_centered = means - means.mean(axis=0, keepdims=True)
            slopes = (x_centered[:, None] * y_centered).sum(axis=0) / denom

            all_data['slope'].extend(slopes.tolist())
            all_data['depth_bin'].extend([b] * len(head_indices))
            all_data['model'].extend([MODEL_PLOT_CONFIG[model]['name']] * len(head_indices))

    df = pd.DataFrame(all_data)

    fig, ax = plt.subplots(figsize=(24, 6))
    sns.violinplot(
        data=df, x='depth_bin', y='slope', hue='model',
        palette={MODEL_PLOT_CONFIG[m]['name']: MODEL_PLOT_CONFIG[m]['color'] for m in MODEL_INFO.keys()},
        ax=ax
    )
    ax.set_xlabel('Relative depth (binned, shared across models)')
    ax.set_ylabel('Attention Entropy Slope')
    ax.set_title(f'Attention Entropy Change vs Perturbation, by Relative Depth ({ptb_type} perturbation)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f'results/{ptb_type}_attention_entropy_all_models.png')
    plt.close(fig)

    return df

def compute_attention_entropy_metrics(ptb_type):
    for model in MODEL_INFO.keys():
        print('Computing attention entropy metrics for model:', model)
        layer_indices = list(range(MODEL_INFO[model]['num_layers']))
        head_indices = list(range(MODEL_INFO[model]['num_heads']))
        ptb_pcts = [x*5 for x in range(11)] if ptb_type != 'shuffle' else [x*5 for x in range(21)]

        entropy_cols = [f'attn_layer{i}_head_{h}_entropy_norm' for i in layer_indices for h in head_indices]
        dfs_by_pct = {}
        for pct in ptb_pcts:
            try:
                seq = pd.read_csv(f'results/{model}/{ptb_type}/{pct}/evals.csv', usecols=entropy_cols)
            except (FileNotFoundError, ValueError):
                seq = pd.read_csv(f'results/{model}/char/0/evals.csv', usecols=entropy_cols)
            dfs_by_pct[pct] = seq

        ptb_arr = np.array(ptb_pcts, dtype=float)
        x_mean = ptb_arr.mean()
        x_centered = ptb_arr - x_mean
        denom = (x_centered ** 2).sum()

        plot_dat = {'layer': [], 'slope': []}
        for i in layer_indices:
            means = np.array([
                [dfs_by_pct[pct][f'attn_layer{i}_head_{h}_entropy_norm'].mean() for h in head_indices]
                for pct in ptb_pcts
            ])
            y_centered = means - means.mean(axis=0, keepdims=True)
            slopes = (x_centered[:, None] * y_centered).sum(axis=0) / denom  # vectorized linregress slope
            plot_dat['slope'].extend(slopes.tolist())
            plot_dat['layer'].extend([i] * len(head_indices))

        fig, ax = plt.subplots(figsize=(10, 3))
        sns.violinplot(plot_dat, x='layer', y='slope', palette=[MODEL_PLOT_CONFIG[model]['color']], hue='layer', legend=False, ax=ax)
        ax.set_title(f"{MODEL_PLOT_CONFIG[model]['name']} Attention Entropy Change vs Perturbation Percentage ({ptb_type} perturbation)")
        ax.set_ylabel("Attention Entropy Delta")
        ax.set_xticklabels(layer_indices)
        plt.tight_layout()
        plt.savefig(f'results/{ptb_type}_attention_entropy_distribution_{model}.png')
        plt.close(fig)


def compute_aggregate_output_metric(metric_name, ptb_type):
    ptb_pcts = [x*5 for x in range(11)] if ptb_type != 'shuffle' else [x*5 for x in range(21)]

    data = {x: [] for x in MODEL_INFO.keys()}
    data['ptb_pct'] = ptb_pcts

    for model in MODEL_INFO.keys():
        for pct in ptb_pcts:
            try:
                seq = pd.read_csv(f'results/{model}/{ptb_type}/{pct}/evals.csv')
            except:
                seq = pd.read_csv(f'results/{model}/char/0/evals.csv')
            data[model].append(np.mean(seq[metric_name]))
    
    df = pd.DataFrame(data)
    df.to_csv(f'results/{ptb_type}_aggregate_{metric_name}_no_entropy.csv', index=False)

    plt.figure(figsize=(10, 6))
    for model in MODEL_INFO.keys():
        plt.plot(df['ptb_pct'], df[model], label=MODEL_PLOT_CONFIG[model]['name'], color=MODEL_PLOT_CONFIG[model]['color'])
    plt.xlabel('Perturbation Percentage')
    plt.ylabel(metric_name)
    plt.title(f'Aggregate {metric_name} vs Perturbation Percentage for {ptb_type} Perturbation')
    plt.legend()
    plt.savefig(f'results/{ptb_type}_aggregate_{metric_name}.png')
    plt.close()

if __name__ == "__main__":
    for ptb_type in ['shuffle', 'token', 'char']:
        # for metric in ['nll', 'output_divergence']:
        #     print(f"Computing aggregate metrics for metric: {metric}, perturbation type: {ptb_type}")
        #     compute_aggregate_output_metric(metric, ptb_type)
        compute_attention_entropy_metrics_comparison(ptb_type)