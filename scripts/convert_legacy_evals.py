"""
Convert legacy evals.csv files (old per-head attention-entropy schema) to the
lean schema written by the fixed src/eval.py:

  * 1,200 per-head attention columns (48 layers x 25 heads for gpt2-xl)
    -> 48 per-layer mean columns ``attn_entropy_layer_{i}``
  * float64 -> float32 (halves CSV size)
  * inf/nan intrinsic-dim values (from the old degenerate-division bug)
    -> 0.0, matching the fixed estimators

Run this ONCE before resuming with run_cross_model.py so the surviving
gpt2-xl results (char/token/word complete, synonym partial) match the schema
produced by the fixed eval loop. Files already in the new schema are skipped.

Usage:
    python scripts/convert_legacy_evals.py                      # all models in results_v2
    python scripts/convert_legacy_evals.py --models gpt2-xl     # only gpt2-xl
    python scripts/convert_legacy_evals.py --out-root results_v2 --dry-run
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ATTN_COL_RE = re.compile(r'^attn_layer(\d+)_head_\d+_entropy_norm$')

BASE_COLS = ['sample', 'nll', 'nll_base', 'output_divergence']


def canonical_column_order(df):
    """Exact column order produced by src.eval.eval_loop (post-fix).

    Returns (order_without_attn, attn_layer_indices). ``attn_entropy_layer_{i}``
    columns are inserted by the caller right before ``logit_kl``.
    """
    n_hidden = sum(1 for c in df.columns if c.startswith('activation_cos_sim_layer_'))
    n_attn = len({int(ATTN_COL_RE.match(c).group(1)) for c in df.columns if ATTN_COL_RE.match(c)})
    cols = list(BASE_COLS)
    cols += [f'activation_cka_layer_{i}' for i in range(n_hidden)]
    cols += ['token_in_sample']
    for i in range(n_hidden):
        cols += [f'activation_cos_sim_layer_{i}', f'activation_l2_dist_layer_{i}']
    cols += [f'cka_layer_{i}' for i in range(n_hidden)]
    for i in range(n_hidden):
        cols += [f'intrinsic_dim_clean_layer_{i}', f'intrinsic_dim_perturbed_layer_{i}',
                 f'intrinsic_dim_change_layer_{i}']
    for i in range(n_hidden):
        cols += [f'intrinsic_dim_mknn_clean_layer_{i}', f'intrinsic_dim_mknn_perturbed_layer_{i}',
                 f'intrinsic_dim_mknn_change_layer_{i}']
    cols += ['logit_kl']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'missing columns: {missing[:5]}')
    return cols, list(range(n_attn))


def convert_file(path, dry_run=False):
    print(f'reading {path}', flush=True)
    df = pd.read_csv(path, dtype=np.float32)
    attn_cols = [c for c in df.columns if ATTN_COL_RE.match(c)]
    if not attn_cols:
        print('  already new schema, skipping', flush=True)
        return False
    layers = sorted({int(ATTN_COL_RE.match(c).group(1)) for c in attn_cols})
    layer_means = {}
    for i in layers:
        head_cols = [c for c in attn_cols if c.startswith(f'attn_layer{i}_head_')]
        layer_means[f'attn_entropy_layer_{i}'] = df[head_cols].mean(axis=1).astype(np.float32)

    target_order, _ = canonical_column_order(df)
    df = df.drop(columns=attn_cols)
    df = df[target_order]
    for i in layers:
        df.insert(df.columns.get_loc('logit_kl'), f'attn_entropy_layer_{i}',
                  layer_means[f'attn_entropy_layer_{i}'])

    for c in df.columns:
        if 'intrinsic_dim' in c:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        if c.startswith('attn_entropy_layer_'):
            # legacy fp16 gpt2-xl per-head entropy is all-NaN (clamp/log
            # underflow bug); replace with 0.0 so files stay clean. New runs
            # compute valid entropy in float32.
            df[c] = df[c].fillna(0.0).astype(np.float32)
    if 'sample' in df.columns:
        df['sample'] = df['sample'].astype('int64')
    if 'token_in_sample' in df.columns:
        df['token_in_sample'] = df['token_in_sample'].astype('int64')

    old_size_mb = os.path.getsize(path) / 1e6
    if dry_run:
        print(f'  would convert: {len(attn_cols)} attn cols -> {len(layers)} layer means, '
              f'~{old_size_mb:.0f} MB -> ~{old_size_mb * len(layers) / len(attn_cols):.1f} MB (est.)', flush=True)
        return True
    df.to_csv(path, index=False, float_format='%.6f')
    print(f'  converted: {len(attn_cols)} attn cols -> {len(layers)} layer means, '
          f'{old_size_mb:.0f} MB -> {os.path.getsize(path) / 1e6:.1f} MB', flush=True)
    return True


def find_evals(out_root, models):
    for model in models:
        model_dir = os.path.join(out_root, model)
        if not os.path.isdir(model_dir):
            continue
        for ptb_type in sorted(os.listdir(model_dir)):
            type_dir = os.path.join(model_dir, ptb_type)
            if not os.path.isdir(type_dir):
                continue
            for pct in sorted(os.listdir(type_dir)):
                path = os.path.join(type_dir, pct, 'evals.csv')
                if os.path.isfile(path):
                    yield model, ptb_type, pct, path


def main():
    parser = argparse.ArgumentParser(description='Convert legacy per-head attention CSVs to lean schema')
    parser.add_argument('--out-root', default='results_v2')
    parser.add_argument('--models', default=None, help='Comma-separated model keys (default: all in out_root)')
    parser.add_argument('--dry-run', action='store_true', help='Report what would be converted without writing')
    args = parser.parse_args()

    if not os.path.isdir(args.out_root):
        print(f'out-root not found: {args.out_root}', file=sys.stderr)
        sys.exit(1)

    models = [m.strip() for m in args.models.split(',')] if args.models else \
        [d for d in sorted(os.listdir(args.out_root)) if os.path.isdir(os.path.join(args.out_root, d))]

    n_converted = 0
    for model, ptb_type, pct, path in find_evals(args.out_root, models):
        try:
            if convert_file(path, dry_run=args.dry_run):
                n_converted += 1
        except Exception as e:
            print(f'  !!! failed {path}: {e}', file=sys.stderr)
    print(f'done: {n_converted} file(s) {"would be" if args.dry_run else ""} converted')


if __name__ == '__main__':
    main()
