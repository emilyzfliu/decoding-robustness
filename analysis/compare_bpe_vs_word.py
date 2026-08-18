"""
Compare BPE token substitution vs word-level substitution
using CKA, TwoNN, MKNN, and other evaluation metrics.

Usage: python compare_bpe_vs_word.py [--seed 1] [--debug]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import argparse
import os
import pandas as pd
from tqdm import tqdm

from src.perturbs import perturb
from src.eval import eval_loop


def run_comparison(args):
    SEQ_LEN = 128 if not args.debug else 5
    rng = random.Random(args.seed)
    
    # Setup model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        attn_implementation='eager'
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]
    texts = random.Random(1).sample(texts, 10 if args.debug else 100)
    
    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet consectetur adipiscing elit',
            'Hello world! Hello universe? This is a test sequence.'
        ]
    
    BATCH_SIZE = 4
    percentages = [5, 10, 25, 50] if not args.debug else [10, 25]
    perturbation_types = ['token', 'word']
    
    for ptb_type in perturbation_types:
        for pct in percentages:
            print(f"\nRunning {ptb_type} @ {pct}%")
            out_dir = f'results/compare_bpe_word/{ptb_type}/{pct}'
            os.makedirs(out_dir, exist_ok=True)
            
            texts_perturbed = perturb(texts, pct, rng, ptb_type, tokenizer)
            
            # Resume support
            try:
                seen = set(pd.read_csv(f'{out_dir}/evals.csv')['sample'])
            except:
                seen = set()
            
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={pct}"):
                batch_texts = texts[i:i+BATCH_SIZE]
                batch_perturbed = texts_perturbed[i:i+BATCH_SIZE]
                
                inputs = tokenizer(
                    batch_texts, return_tensors="pt",
                    truncation=True, max_length=SEQ_LEN, padding='max_length'
                )
                inputs_ptb = tokenizer(
                    batch_perturbed, return_tensors="pt",
                    truncation=True, max_length=SEQ_LEN, padding='max_length'
                )
                
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
                    outputs_ptb = model(**inputs_ptb, output_hidden_states=True, output_attentions=True)
                
                res = eval_loop(inputs, outputs, inputs_ptb, outputs_ptb, tokenizer, i)
                res = res[~res['sample'].isin(seen)]
                
                res.to_csv(f'{out_dir}/evals.csv',
                          mode='a', header=(i == 0 and len(seen) == 0), index=False)
                
                del outputs, outputs_ptb
    
    print("\nDone! Results saved to results/compare_bpe_word/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BPE vs word-level token substitution")
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true', help="Use small test data")
    args = parser.parse_args()
    
    print('Running comparison with', args)
    run_comparison(args)