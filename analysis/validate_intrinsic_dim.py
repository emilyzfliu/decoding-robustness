"""
Statistical validation of the TwoNN / MKNN intrinsic-dimension estimators.

Runs a small forward pass on a handful of sequences (clean + perturbed) and
checks estimator behaviour:
  - convergence as the number of subsampled points grows,
  - variance across random subsample seeds,
  - bootstrap confidence intervals at a fixed sample budget.

Usage: python validate_intrinsic_dim.py [--model gpt2] [--n-seq 8] [--ptb-type char] [--ptb-pct 25]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

from src.perturbs import perturb
from src.eval import estimate_intrinsic_dim_2nn, estimate_intrinsic_dim_mknn
from config import MODEL_INFO

LAYERS = [0, 1, 4, 8, 12]
GRID = [100, 250, 500, 1000]
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_B = 200


def collect_hidden(model, tokenizer, texts, max_len=128):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True,
                       max_length=max_len, padding='max_length')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # (n_layers, n_seq, max_len, d) -> points per layer
    hs = out.hidden_states
    return [h[:, :-1, :].reshape(-1, h.shape[-1]).float().cpu().numpy() for h in hs]


def bootstrap_ci(points, estimator, n_use, n_boot, seed=0):
    rng = np.random.RandomState(seed)
    n_total = points.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n_total, size=n_use)
        v = estimator(points[idx], n_samples=n_use, n_use=n_use, seed=seed)
        if v is not None:
            vals.append(v)
    vals = np.array(vals)
    if len(vals) < 10:
        return np.nan, np.nan, np.nan
    return vals.mean(), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--n-seq', type=int, default=8)
    parser.add_argument('--ptb-type', default='char')
    parser.add_argument('--ptb-pct', type=int, default=25)
    parser.add_argument('--out-dir', default='figures')
    args = parser.parse_args()

    model_name = MODEL_INFO[args.model]['model_name']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager').to(device)
    model.eval()

    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
    texts = [t for t in ds['test']['text'] if len(t.split()) > 128]
    rng = random.Random(1)
    texts = rng.sample(texts, args.n_seq)

    rng_ptb = random.Random(1)
    texts_ptb = perturb(texts, args.ptb_pct, rng_ptb, args.ptb_type, tokenizer)

    print(f'Collecting hidden states: {args.model}, {args.n_seq} seqs, '
          f'{args.ptb_type}@{args.ptb_pct}%')
    pts_clean = collect_hidden(model, tokenizer, texts)
    pts_ptb = collect_hidden(model, tokenizer, texts_ptb)

    estimators = {
        '2NN': estimate_intrinsic_dim_2nn,
        'MKNN': estimate_intrinsic_dim_mknn,
    }

    # --- convergence + seed-stability ---
    fig, axes = plt.subplots(len(LAYERS), 2, figsize=(14, 3 * len(LAYERS)), sharex=True)
    fig.suptitle(f'Intrinsic-dim estimator convergence ({args.model}, {args.ptb_type}@{args.ptb_pct}%)',
                 fontsize=13)

    summary_rows = []
    for L in LAYERS:
        for j, (name, est) in enumerate(estimators.items()):
            ax = axes[LAYERS.index(L), j]
            for setting, pts in [('clean', pts_clean[L]), ('perturbed', pts_ptb[L])]:
                means, stds = [], []
                for n in GRID:
                    n_use = min(n, pts.shape[0])
                    vals = []
                    for s in SEEDS:
                        v = est(pts, n_samples=n_use, n_use=n_use, seed=s)
                        if v is not None:
                            vals.append(v)
                    vals = np.array(vals)
                    means.append(vals.mean() if len(vals) else np.nan)
                    stds.append(vals.std() if len(vals) else np.nan)
                means = np.array(means); stds = np.array(stds)
                ax.plot(GRID, means, 'o-', label=setting, linewidth=2)
                ax.fill_between(GRID, means - stds, means + stds, alpha=0.25)
                ax.axhline(means[-1], color='gray', lw=0.6, ls='--')
            ax.set_title(f'Layer {L} - {name}')
            ax.set_xlabel('Subsample size (n)')
            ax.set_ylabel('Intrinsic dim')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_conv = os.path.join(args.out_dir, 'intrinsic_dim_convergence.png')
    plt.savefig(out_conv, bbox_inches='tight', dpi=150)
    print(f'Saved {out_conv}')

    # --- bootstrap CI at fixed budget ---
    n_use = 500
    print('=' * 70)
    print(f'BOOTSTRAP CI (B={BOOTSTRAP_B}, n={n_use} points) - mean [2.5%, 97.5%]')
    print('=' * 70)
    for L in LAYERS:
        for name, est in estimators.items():
            for setting, pts in [('clean', pts_clean[L]), ('perturbed', pts_ptb[L])]:
                m, lo, hi = bootstrap_ci(pts, est, n_use, BOOTSTRAP_B)
                summary_rows.append({'layer': L, 'estimator': name, 'setting': setting,
                                     'mean': m, 'ci_lo': lo, 'ci_hi': hi})
                print(f'  L{L:>2d} {name:>4s} {setting:<10} {m:7.2f} [{lo:7.2f}, {hi:7.2f}]')

    df = pd.DataFrame(summary_rows)
    out_csv = os.path.join(args.out_dir, 'intrinsic_dim_validation_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f'Saved {out_csv}')

    print('\nInterpretation:')
    print('  - If mean changes little with n, the estimator has converged;')
    print('  - Narrow CI relative to the gap clean-vs-perturbed => the effect is statistically reliable.')


if __name__ == '__main__':
    main()
