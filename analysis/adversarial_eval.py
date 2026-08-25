"""
Adversarial / injection perturbation experiments.

Two experiments, run independently or together:

1. Context insertion (--experiment context_insertion): inject irrelevant /
   misleading / adversarially-optimized text into a passage; measure
   perplexity and next-token accuracy. Conditions: clean, topic_shift,
   misleading_claim, adversarial.

2. Question-level perturbations (--experiment question_level): rephrase or
   adversarially attack a passage's context while holding out its final word,
   then check via loose substring match whether the model's greedy
   continuation contains that word. Conditions: clean, synonym, reorder,
   negation_paraphrase, adversarial_swap.

See ADVERSARIAL_METHODOLOGY.md for caveats that apply to interpreting the
results (in particular: the question-level task/metric mismatch for the
adversarial_swap condition, absence of confidence intervals across seeds, and
the proxy-task-vs-real-QA-benchmark distinction).
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from time import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_INFO
from src.adversarial_perturbs import context_insertion, question_perturbation
from src.eval import nll, next_token_accuracy, perplexity

CONTEXT_CONDITIONS = ['clean', 'topic_shift', 'misleading_claim', 'adversarial']
QUESTION_CONDITIONS = ['clean', 'synonym', 'reorder', 'negation_paraphrase', 'adversarial_swap']


def load_model_and_data(args):
    model_name = MODEL_INFO[args.model]['model_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=MODEL_INFO[args.model]['attn_implementation']
        )
    except TypeError:
        # Older transformers versions don't accept attn_implementation.
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    seq_len = 128 if not args.debug else 5
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > seq_len]

    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.',
            'Hello world! This is a test passage written in the year 1995 by John Smith.',
        ]
    else:
        texts = texts[:args.n_samples]

    return model, tokenizer, device, texts


def run_context_insertion(args, model, tokenizer, device, texts):
    rng = random.Random(args.seed)
    results_dir = f'results/{args.model}/adversarial/context_insertion_{args.max_length}tok'
    os.makedirs(results_dir, exist_ok=True)

    summary_rows = []
    clean_ppl = None

    for condition in CONTEXT_CONDITIONS:
        cond_dir = f'{results_dir}/{condition}'
        os.makedirs(cond_dir, exist_ok=True)

        perturbed_texts = context_insertion(
            texts, condition, rng, tokenizer, model=model, device=device, distractor_pool=texts,
            max_length=args.max_length,
        )

        try:
            seen = set(pd.read_csv(f'{cond_dir}/evals.csv')['sample'])
        except Exception:
            seen = set()

        rows = []
        for i, text in enumerate(perturbed_texts):
            if i in seen:
                continue
            # No padding: this loop is always batch-size-1, and padding to a fixed
            # length combined with left-padding pushes real content to very high
            # position indices, badly degrading GPT-2's predictions.
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=args.max_length).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            rows.append({
                'sample': i,
                'nll': nll(inputs, outputs)[0],
                'perplexity': perplexity(inputs, outputs)[0],
                'next_token_acc': next_token_accuracy(inputs, outputs)[0],
            })

        if rows:
            pd.DataFrame(rows).to_csv(
                f'{cond_dir}/evals.csv', mode='a', header=(len(seen) == 0), index=False
            )

        full_df = pd.read_csv(f'{cond_dir}/evals.csv')
        corpus_ppl = float(np.exp(full_df['nll'].mean()))
        acc = float(full_df['next_token_acc'].mean())
        if condition == 'clean':
            clean_ppl = corpus_ppl
        summary_rows.append({
            'condition': condition,
            'ppl': corpus_ppl,
            'delta_ppl_vs_clean': corpus_ppl - clean_ppl if clean_ppl is not None else 0.0,
            'next_token_acc': acc,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{results_dir}/summary.csv', index=False)
    print("\nContext Insertion summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def _split_context_and_target(text):
    words = text.split()
    # WikiText-2 raw text space-separates punctuation (e.g. "Hall ."), so the
    # literal last token is frequently pure punctuation — walk back to the
    # last token that actually contains a word character.
    end = len(words)
    while end > 0 and not re.search(r'\w', words[end - 1]):
        end -= 1
    if end < 2:
        return None, None
    context = ' '.join(words[:end - 1])
    target = re.sub(r'[^\w]', '', words[end - 1]).lower()
    if not target:
        return None, None
    return context, target


def run_question_level(args, model, tokenizer, device, texts):
    rng = random.Random(args.seed)
    results_dir = f'results/{args.model}/adversarial/question_level_{args.max_length}tok'
    os.makedirs(results_dir, exist_ok=True)

    pairs = [_split_context_and_target(t) for t in texts]
    pairs = [(c, t) for c, t in pairs if c and t]
    contexts = [c for c, _ in pairs]
    targets = [t for _, t in pairs]

    summary_rows = []
    for condition in QUESTION_CONDITIONS:
        cond_dir = f'{results_dir}/{condition}'
        os.makedirs(cond_dir, exist_ok=True)

        perturbed_contexts = question_perturbation(
            contexts, condition, rng, tokenizer, model=model, device=device,
            max_length=args.max_length,
        )

        try:
            seen = set(pd.read_csv(f'{cond_dir}/evals.csv')['sample'])
        except Exception:
            seen = set()

        rows = []
        for i, (ctx, target) in enumerate(zip(perturbed_contexts, targets)):
            if i in seen:
                continue
            inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            continuation = tokenizer.decode(
                gen[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )
            match = target in continuation.lower()
            rows.append({'sample': i, 'target': target, 'continuation': continuation, 'match': match})

        if rows:
            pd.DataFrame(rows).to_csv(
                f'{cond_dir}/evals.csv', mode='a', header=(len(seen) == 0), index=False
            )

        full_df = pd.read_csv(f'{cond_dir}/evals.csv')
        acc = float(full_df['match'].mean())
        summary_rows.append({'condition': condition, 'accuracy': acc})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f'{results_dir}/summary.csv', index=False)
    print("\nQuestion-Level Perturbation summary:")
    print(summary_df.to_string(index=False))
    return summary_df


def main(args):
    start_time = time()
    model, tokenizer, device, texts = load_model_and_data(args)

    if args.experiment in ('context_insertion', 'both'):
        run_context_insertion(args, model, tokenizer, device, texts)
    if args.experiment in ('question_level', 'both'):
        run_question_level(args, model, tokenizer, device, texts)

    print(f"Total time taken: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial/injection perturbation experiments")
    parser.add_argument("--model", help="Model name (key in config.MODEL_INFO)", type=str, default='gpt2')
    parser.add_argument("--experiment", type=str, default='both',
                         choices=['context_insertion', 'question_level', 'both'])
    parser.add_argument("--n-samples", type=int, default=50,
                         help="Subset size — adversarial/HotFlip conditions do per-sample gradient "
                              "optimization, so full-dataset runs aren't practical")
    parser.add_argument("--max-length", type=int, default=1024,
                         help="Tokenizer truncation length. Results are written under a "
                              "{max_length}tok-suffixed directory so runs at different lengths "
                              "never overwrite each other.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()
    print('Running with', args)
    main(args)
