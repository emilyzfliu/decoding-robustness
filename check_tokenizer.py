"""
Sanity-check the "128 tokens, ~no padding" assumption for any model's tokenizer
(the eval math assumes full, unpadded 128-token sequences).

Usage: python check_tokenizer.py --model facebook/opt-6.7b
"""
import argparse, random
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
texts = [x for x in ds["test"]["text"] if len(x.split()) > 128]
texts = random.Random(1).sample(texts, 100)          # same sampling as main.py

lens = np.array([len(tok(t)["input_ids"]) for t in texts])
enc = tok(texts, truncation=True, max_length=128, padding="max_length")
pads = np.array([sum(1 for m in am if m == 0) for am in enc["attention_mask"]])

print(f"model = {args.model}")
print(f"  bos_token_id={tok.bos_token_id}  first 3 tokens of sample0: {tok(texts[0])['input_ids'][:3]}")
print(f"  raw token counts: min={lens.min()} median={int(np.median(lens))} max={lens.max()}")
print(f"  passages < 128 tokens (would need padding): {(lens < 128).sum()}/100")
print(f"  after truncation to 128 -> pad/seq: mean={pads.mean():.1f} max={pads.max()} | seqs with padding: {(pads > 0).sum()}/100")
print("  => OK (no padding, assumption holds)" if (pads > 0).sum() == 0 else "  => WARNING: padding present, revisit eval masking")
