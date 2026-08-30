"""
Measures how much of HotFlip's edit budget actually gets used.

hotflip_token_substitution's --ptb-pct sets a CEILING on the number of
edited positions (n_to_replace = floor(pct * n_tokens / 100)) — a position
is only actually changed if one of the 50 gradient-shortlisted candidates
raises the sequence's own loss; otherwise it's left as-is. So the realized
edit count is <= the ceiling, and this script measures how close to that
ceiling HotFlip actually gets in practice, for the appendix.

Usage:
    python analysis/measure_hotflip_budget_usage.py --model gpt2 --pcts 5,30,50 --n-samples 30
    python analysis/measure_hotflip_budget_usage.py --model gpt2-xl --pcts 5,30,50 --n-samples 30 --out results_budget/gpt2-xl.csv
    # n_candidates sensitivity sweep at a single pct:
    python analysis/measure_hotflip_budget_usage.py --model gpt2 --pcts 30 --n-candidates 1,5,10,20,50,100 --n-samples 30
"""
import argparse
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_INFO
from src.hotflip import hotflip_attack


def load_model_and_texts(model_key, n_samples, seed=1, seq_len=128, max_length=128):
    model_name = MODEL_INFO[model_key]['model_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=MODEL_INFO[model_key]['attn_implementation']
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > seq_len]
    texts = random.Random(seed).sample(texts, min(n_samples, len(texts)))

    return model, tokenizer, device, texts


def measure_one(text, pct, tokenizer, model, device, max_length=128, n_candidates=50):
    input_ids_orig = tokenizer(text, add_special_tokens=False, truncation=True,
                                max_length=max_length)['input_ids']
    n_tokens = len(input_ids_orig)
    if n_tokens < 2:
        return None
    n_to_replace = min(n_tokens, max(1, int(pct * n_tokens / 100)))

    ids_tensor = torch.tensor(input_ids_orig, dtype=torch.long)
    attention_mask = torch.ones_like(ids_tensor)
    rng = random.Random(0)  # local rng for the attack's position ordering; doesn't need to match main.py's

    attacked_ids = hotflip_attack(
        model, tokenizer, ids_tensor, attention_mask, list(range(n_tokens)),
        device, rng, n_candidates=n_candidates, n_iters=n_to_replace,
    )
    realized = int((attacked_ids != ids_tensor).sum().item())
    changed_positions = torch.nonzero(attacked_ids != ids_tensor, as_tuple=True)[0].tolist()
    return {
        'n_tokens': n_tokens,
        'ceiling': n_to_replace,
        'realized': realized,
        'ratio': realized / n_to_replace if n_to_replace else float('nan'),
        'hit_ceiling': realized == n_to_replace,
        'original_text': text,
        'perturbed_text': tokenizer.decode(attacked_ids),
        'original_ids': input_ids_orig,
        'perturbed_ids': attacked_ids.tolist(),
        'changed_positions': changed_positions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--pcts', default='5,30,50')
    parser.add_argument('--n-samples', type=int, default=30)
    parser.add_argument('--n-candidates', default='50',
                         help='Gradient-shortlist size per attacked position. Comma-separated for a '
                              'sensitivity sweep, e.g. 1,5,10,20,50,100')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default=None, help='Optional path to write per-sample rows as CSV')
    args = parser.parse_args()

    pcts = [int(p) for p in args.pcts.split(',')]
    n_candidates_list = [int(k) for k in args.n_candidates.split(',')]
    model, tokenizer, device, texts = load_model_and_texts(args.model, args.n_samples, seed=args.seed)
    print(f'{args.model}: {len(texts)} texts, device={device}')

    rows = []
    for pct in pcts:
        for n_cand in n_candidates_list:
            t0 = time.time()
            for i, text in enumerate(texts):
                m = measure_one(text, pct, tokenizer, model, device, n_candidates=n_cand)
                if m is None:
                    continue
                m.update({'model': args.model, 'pct': pct, 'n_candidates': n_cand, 'sample': i})
                rows.append(m)
            dt = time.time() - t0
            grp_rows = [r for r in rows if r['pct'] == pct and r['n_candidates'] == n_cand]
            mean_ratio = sum(r['ratio'] for r in grp_rows) / len(grp_rows)
            hit_frac = sum(r['hit_ceiling'] for r in grp_rows) / len(grp_rows)
            print(f'  pct={pct:3d} n_candidates={n_cand:4d}: mean realized/ceiling = {mean_ratio:.3f}, '
                  f'hit ceiling on {hit_frac*100:.1f}% of samples  ({dt:.1f}s)')

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f'wrote {args.out}')

    print('\nSummary by (pct, n_candidates):')
    summary = df.groupby(['pct', 'n_candidates']).agg(
        mean_ceiling=('ceiling', 'mean'),
        mean_realized=('realized', 'mean'),
        mean_ratio=('ratio', 'mean'),
        hit_ceiling_frac=('hit_ceiling', 'mean'),
    )
    print(summary.to_string())


if __name__ == '__main__':
    main()
