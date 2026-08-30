import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import argparse
import json
import os
import subprocess
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
from time import time

from src.perturbs import perturb
from src.eval import eval_loop
from config import MODEL_INFO


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_manifest(pct_dir, args):
    """Provenance for this results directory: what code/config produced it.
    Overwritten each run -- meant to reflect the *latest* run, not history."""
    manifest = {
        'git_commit': _git_commit(),
        'dtype': MODEL_INFO[args.model]['dtype'],
        'seed': args.seed,
        'n_candidates': args.n_candidates,
        'cli_args': vars(args),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with open(f'{pct_dir}/meta.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def write_texts_csv(pct_dir, texts, texts_perturbed, ptb_type, tokenizer, max_length=128):
    """Original/perturbed text per sample, plus (for hotflip) the realized edit count
    and positions -- none of this is derivable from evals.csv alone."""
    rows = []
    for i, (orig, pert) in enumerate(zip(texts, texts_perturbed)):
        row = {'sample': i, 'original_text': orig, 'perturbed_text': pert}
        # Realized edit count is well-defined for any perturbation that preserves token
        # count (hotflip, token, shuffle by construction) -- not for char/typo/word/synonym,
        # which may shift token count under retokenization (same caveat CKA already has).
        # Left blank rather than guessed at for those, rather than hardcoding by ptb_type.
        orig_ids = tokenizer(orig, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']
        pert_ids = tokenizer(pert, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']
        if len(orig_ids) == len(pert_ids):
            changed = [j for j, (a, b) in enumerate(zip(orig_ids, pert_ids)) if a != b]
            row['n_tokens_changed'] = len(changed)
            row['changed_positions'] = changed
        rows.append(row)
    pd.DataFrame(rows).to_csv(f'{pct_dir}/texts.csv', index=False)

def run(args):
    start_time = time()
    SEQ_LEN = 128 if not args.debug else 5

    model_name = MODEL_INFO[args.model]['model_name']
        
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up models

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if MODEL_INFO[args.model]['dtype'] == 'fp16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      attn_implementation=MODEL_INFO[args.model]['attn_implementation'],
      torch_dtype=dtype
    ).to(device)

    model.eval()

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]
    if args.n_samples > 0 and not args.debug:
        texts = random.Random(1).sample(texts, args.n_samples)
    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet',
            'Hello world! Hello universe?'
        ]
    
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else MODEL_INFO[args.model]['max_batch_size']

    ptb_type = args.ptb_type

    if args.ptb_pct != -1:
        ptb_pcts = [args.ptb_pct]
    else:
        if ptb_type == 'char':
            ptb_pcts = [x*5 for x in range(0, 11)] # baseline [hacky]
        else:
            ptb_pcts = [x*5 for x in range(1, 11)]
    
    # A non-default n_candidates only matters for hotflip, and changes the results it
    # produces -- give it its own results directory so it never mixes with (or silently
    # overwrites) the default-50 data already collected.
    pct_dirname_suffix = f'_nc{args.n_candidates}' if (ptb_type == 'hotflip' and args.n_candidates != 50) else ''

    for ptb_pct in ptb_pcts:

        texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer, model=model, device=device,
                                   n_candidates=args.n_candidates)

        pct_dir = f'{args.out_root}/{args.model}/{ptb_type}/{ptb_pct}{pct_dirname_suffix}'
        os.makedirs(pct_dir, exist_ok=True)

        if not args.debug:
            write_manifest(pct_dir, args)
            write_texts_csv(pct_dir, texts, texts_perturbed, ptb_type, tokenizer)

        try:
            seen = set(pd.read_csv(f'{pct_dir}/evals.csv', usecols=['sample'])['sample'])
        except:
            seen = set()

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={ptb_pct}"):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_texts_perturbed = texts_perturbed[i:i+BATCH_SIZE]
            
            inputs = tokenizer(batch_texts, return_tensors="pt", 
                            truncation=True, max_length=128, padding='max_length').to(device)
            inputs_perturbed = tokenizer(batch_texts_perturbed, return_tensors="pt",
                                        truncation=True, max_length=128, padding='max_length').to(device)
            
            eval_hidden = MODEL_INFO[args.model]['eval_hidden_states']
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states= eval_hidden, output_attentions=eval_hidden)
                outputs_perturbed = model(**inputs_perturbed, output_hidden_states=eval_hidden, output_attentions=eval_hidden)
            
            res = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, i, output_only=(not eval_hidden))

            res = res[~res['sample'].isin(seen)]
            if not args.debug:
                res.to_csv(f'{pct_dir}/evals.csv',
                                        mode='a', header=(i==0 and len(seen) == 0), index=False, float_format='%.6f')
            else:
                res.to_csv(f'{args.out_root}/{args.model}/debug.csv', mode='a', header=(i==0 and len(seen) == 0), index=False, float_format='%.6f')
            
            del outputs, outputs_perturbed
    print(f"Total time taken: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--model", help="Model name", type=str, default='gpt2')
    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle']", type=str, default='char')
    parser.add_argument("--ptb-pct", help="Percent of input text perturbed", type=int, default=-1)
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--n-samples", help="Number of sequences to sample (-1 = use all)", type=int, default=-1)
    parser.add_argument("--batch-size", help="Batch size (<=0 = per-model max_batch_size)", type=int, default=0)
    parser.add_argument("--out-root", help="Root directory for results (default: results)", type=str, default='results')
    parser.add_argument("--n-candidates", help="HotFlip gradient-shortlist size (only affects ptb-type=hotflip); "
                                                "non-default values get their own results dir suffix", type=int, default=50)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running with', args)

    run(args)