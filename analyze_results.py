"""
Analyze BPE vs Word comparison results.
"""
import pandas as pd
import numpy as np

print('=' * 80)
print('BPE vs WORD SUBSTITUTION COMPARISON RESULTS')
print('=' * 80)

# --- CKA Comparison ---
print('\n--- CKA (1 = identical, 0 = unrelated) ---')
print(f"{'Pct':>5}  {'Token (BPE)':>15}  {'Word-Level':>15}  {'Diff':>15}")
print('-' * 55)

for pct in [5, 10, 25, 50]:
    bpe_vals, word_vals = [], []
    for layer in range(12):
        try:
            df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
            df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
            bpe_vals.append(df_bpe[f'cka_layer_{layer}'].mean())
            word_vals.append(df_word[f'cka_layer_{layer}'].mean())
        except FileNotFoundError:
            bpe_vals.append(0)
            word_vals.append(0)
    
    bpe_mean = np.mean(bpe_vals)
    word_mean = np.mean(word_vals)
    print(f'{pct:>5}  {bpe_mean:>15.4f}  {word_mean:>15.4f}  {bpe_mean - word_mean:>15.4f}')

# --- TwoNN Intrinsic Dim Change ---
print('\n--- TwoNN Intrinsic Dim Change (positive = expansion) ---')
print(f"{'Pct':>5}  {'Token (BPE)':>15}  {'Word-Level':>15}  {'Diff':>15}")
print('-' * 55)

for pct in [5, 10, 25, 50]:
    try:
        df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
        df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
        
        cols_bpe = [c for c in df_bpe.columns if c.startswith('intrinsic_dim_change_layer_')]
        cols_word = [c for c in df_word.columns if c.startswith('intrinsic_dim_change_layer_')]
        
        bpe_mean = np.mean([df_bpe[c].mean() for c in cols_bpe])
        word_mean = np.mean([df_word[c].mean() for c in cols_word])
        print(f'{pct:>5}  {bpe_mean:>15.4f}  {word_mean:>15.4f}  {bpe_mean - word_mean:>15.4f}')
    except FileNotFoundError:
        print(f'{pct:>5}  {"N/A":>15}  {"N/A":>15}  {"N/A":>15}')

# --- MKNN Intrinsic Dim Change ---
print('\n--- MKNN Intrinsic Dim Change (positive = expansion) ---')
print(f"{'Pct':>5}  {'Token (BPE)':>15}  {'Word-Level':>15}  {'Diff':>15}")
print('-' * 55)

for pct in [5, 10, 25, 50]:
    try:
        df_bpe = pd.read_csv(f'results/compare_bpe_word/token/{pct}/evals.csv')
        df_word = pd.read_csv(f'results/compare_bpe_word/word/{pct}/evals.csv')
        
        cols_bpe = [c for c in df_bpe.columns if c.startswith('intrinsic_dim_mknn_change_layer_')]
        cols_word = [c for c in df_word.columns if c.startswith('intrinsic_dim_mknn_change_layer_')]
        
        bpe_mean = np.mean([df_bpe[c].mean() for c in cols_bpe])
        word_mean = np.mean([df_word[c].mean() for c in cols_word])
        print(f'{pct:>5}  {bpe_mean:>15.4f}  {word_mean:>15.4f}  {bpe_mean - word_mean:>15.4f}')
    except FileNotFoundError:
        print(f'{pct:>5}  {"N/A":>15}  {"N/A":>15}  {"N/A":>15}')

# --- Per-Layer CKA at 25% ---
print('\n')
print('=' * 80)
print('PER-LAYER CKA at 25% Perturbation')
print('=' * 80)
print(f"{'Layer':>6}  {'Token (BPE)':>15}  {'Word-Level':>15}  {'Diff':>15}")
print('-' * 55)

try:
    df_bpe = pd.read_csv('results/compare_bpe_word/token/25/evals.csv')
    df_word = pd.read_csv('results/compare_bpe_word/word/25/evals.csv')

    for i in range(12):
        cka_bpe = df_bpe[f'cka_layer_{i}'].mean()
        cka_word = df_word[f'cka_layer_{i}'].mean()
        print(f'Layer {i:>2d}:  {cka_bpe:>15.4f}  {cka_word:>15.4f}  {cka_bpe - cka_word:>15.4f}')
except FileNotFoundError:
    print('Data not found. Run compare_bpe_vs_word.py first.')

# --- Summary of columns ---
print('\n')
print('=' * 80)
print('METRICS CAPTURED IN EVALS.CSV')
print('=' * 80)
try:
    df = pd.read_csv('results/compare_bpe_word/token/5/evals.csv')
    print(f'Total columns: {len(df.columns)}')
    print(f'CKA columns: {len([c for c in df.columns if c.startswith("cka_")])}')
    print(f'TwoNN columns: {len([c for c in df.columns if "intrinsic_dim_" in c and "mknn" not in c])}')
    print(f'MKNN columns: {len([c for c in df.columns if "mknn" in c])}')
    print(f'Cos sim columns: {len([c for c in df.columns if "cos_sim" in c])}')
    print(f'L2 columns: {len([c for c in df.columns if "l2_dist" in c])}')
    print(f'Attention entropy columns: {len([c for c in df.columns if "entropy" in c])}')
except FileNotFoundError:
    print('Data not found.')