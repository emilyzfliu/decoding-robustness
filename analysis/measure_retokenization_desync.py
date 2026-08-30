"""
Measures how often, and by how much, decode->re-tokenize round trips desync
token count/position for perturbations that only return decoded text (all of
them -- token_substitution, hotflip_token_substitution, etc. never return
their internal token-id ground truth, only a decoded string).

main.py's eval step re-tokenizes that string independently, and the paper's
CKA methodology (Section 3.3.2) claims 'token' and 'shuffle' preserve exact
token count / position-wise alignment -- this checks whether that's actually
true in practice, not just in principle. Also supports ptb-type=hotflip,
since that's what RQ4 actually depends on.

Usage:
    python analysis/measure_retokenization_desync.py --model gpt2 --ptb-type token --pct 30 --n-samples 100
    python analysis/measure_retokenization_desync.py --model gpt2 --ptb-type hotflip --pct 30 --n-samples 20
"""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_INFO
from src.hotflip import hotflip_attack


def ground_truth_token_substitution(orig_ids, pct, rng, vocab_size):
    return [
        rng.randint(0, vocab_size - 1) if rng.randint(1, 100) < pct else tid
        for tid in orig_ids
    ]


def ground_truth_hotflip(orig_ids, pct, rng, model, tokenizer, device, n_candidates=50):
    n_tokens = len(orig_ids)
    n_to_replace = min(n_tokens, max(1, int(pct * n_tokens / 100)))
    ids_tensor = torch.tensor(orig_ids, dtype=torch.long)
    attention_mask = torch.ones_like(ids_tensor)
    attacked_ids = hotflip_attack(
        model, tokenizer, ids_tensor, attention_mask, list(range(n_tokens)),
        device, rng, n_candidates=n_candidates, n_iters=n_to_replace,
    )
    return attacked_ids.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2')
    parser.add_argument('--ptb-type', default='token', choices=['token', 'hotflip'])
    parser.add_argument('--pct', type=int, default=30)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--n-candidates', type=int, default=50)
    args = parser.parse_args()

    model_name = MODEL_INFO[args.model]['model_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = None
    if args.ptb_type == 'hotflip':
        dtype = torch.float32 if MODEL_INFO[args.model]['dtype'] == 'fp32' else torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=MODEL_INFO[args.model]['attn_implementation'], torch_dtype=dtype
            ).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > 128]
    texts = random.Random(1).sample(texts, min(args.n_samples, len(texts)))

    rng = random.Random(args.seed)
    vocab_size = tokenizer.vocab_size

    length_drifts = []
    n_length_mismatch = 0
    position_mismatch_fracs = []

    for text in texts:
        orig_ids = tokenizer(text, add_special_tokens=False, truncation=True,
                              max_length=args.max_length)['input_ids']
        if len(orig_ids) < 2:
            continue

        if args.ptb_type == 'token':
            new_ids = ground_truth_token_substitution(orig_ids, args.pct, rng, vocab_size)
        else:
            new_ids = ground_truth_hotflip(orig_ids, args.pct, rng, model, tokenizer, device,
                                            n_candidates=args.n_candidates)

        decoded = tokenizer.decode(new_ids)
        reencoded = tokenizer(decoded, truncation=True, max_length=args.max_length)['input_ids']

        drift = len(reencoded) - len(orig_ids)
        length_drifts.append(drift)
        if drift != 0:
            n_length_mismatch += 1
        else:
            mismatches = sum(1 for a, b in zip(new_ids, reencoded) if a != b)
            position_mismatch_fracs.append(mismatches / len(orig_ids))

    n = len(length_drifts)
    print(f'{args.model}, ptb_type={args.ptb_type}, pct={args.pct}, n={n} samples')
    print(f'  length changed after decode->re-encode: {n_length_mismatch}/{n} '
          f'({n_length_mismatch/n*100:.1f}%)')
    if length_drifts:
        avg_abs_drift = sum(abs(d) for d in length_drifts) / len(length_drifts)
        max_abs_drift = max(abs(d) for d in length_drifts)
        print(f'  mean |drift|: {avg_abs_drift:.2f} tokens, max |drift|: {max_abs_drift} tokens')
    if position_mismatch_fracs:
        avg_pos_mismatch = sum(position_mismatch_fracs) / len(position_mismatch_fracs)
        print(f'  among length-MATCHED samples ({len(position_mismatch_fracs)}/{n}): '
              f'mean fraction of positions where re-encoded id != ground-truth new_id '
              f'(should be ~0 if truly aligned): {avg_pos_mismatch*100:.2f}%')


if __name__ == '__main__':
    main()
