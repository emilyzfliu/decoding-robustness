"""
Runs HotFlip and its position-coupled random control TOGETHER, from one
shared generation pass, so the random control is guaranteed to touch exactly
the same positions HotFlip did (same edit count, same locations) rather than
an independently-drawn random subset.

Two things this script does that a plain `main.py` invocation can't:
1. Generates both conditions from ONE shared computation (see
   src/perturbs.py:hotflip_and_coupled_random) -- running them as two
   separate calls, even with the same seed, lets their rng consumption
   drift apart after the first text.
2. Builds model inputs DIRECTLY from the raw token ID lists (manual
   left-padding), never from `tokenizer(decoded_text, ...)`. This matters:
   decoding to text and re-tokenizing for eval does NOT preserve exact
   position alignment even when the underlying IDs were coupled correctly --
   different substituted tokens can trigger different BPE re-merging on
   re-encode. Measured directly on a real sample: ~30 of ~97 changed
   positions silently diverged after a decode->re-encode round trip.

Writes to results/{model}/hotflip_paired/{pct}/ and
results/{model}/token_coupled/{pct}/ -- 'hotflip_paired', not plain
'hotflip', since any pre-existing standalone 'hotflip' data at this
(model, pct) has a different rng trajectory and does NOT share positions
with this run's token_coupled output.

Usage:
    python run_paired_hotflip.py --model gpt2 --ptb-pct 30 --n-samples 300
"""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from time import time

import pandas as pd
import random
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from config import MODEL_INFO
from src.eval import eval_loop
from src.perturbs import hotflip_and_coupled_random


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_manifest(pct_dir, args, ptb_type):
    manifest = {
        'git_commit': _git_commit(),
        'dtype': MODEL_INFO[args.model]['dtype'],
        'seed': args.seed,
        'n_candidates': args.n_candidates,
        'ptb_type': ptb_type,
        'paired_run': True,
        'cli_args': vars(args),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    with open(f'{pct_dir}/meta.json', 'w') as f:
        json.dump(manifest, f, indent=2)


def write_texts_csv(pct_dir, texts, texts_perturbed, ids_base, ids_perturbed):
    """Positions/counts come directly from the true ID lists (the ones actually fed
    to the model), not re-derived by retokenizing the decoded text."""
    rows = []
    for i, (orig, pert, o_ids, p_ids) in enumerate(zip(texts, texts_perturbed, ids_base, ids_perturbed)):
        changed = [j for j, (a, b) in enumerate(zip(o_ids, p_ids)) if a != b]
        rows.append({'sample': i, 'original_text': orig, 'perturbed_text': pert,
                      'n_tokens_changed': len(changed), 'changed_positions': changed})
    pd.DataFrame(rows).to_csv(f'{pct_dir}/texts.csv', index=False)


def build_batch_tensors(id_lists, pad_token_id, max_length, device):
    """Left-pads a batch of variable-length ID lists to max_length and builds the
    matching attention mask -- the direct-from-IDs equivalent of
    tokenizer(texts, padding='max_length', truncation=True) with padding_side='left',
    but without ever routing through decoded text.

    Returns a BatchEncoding (not a plain dict): src/eval.py's nll/output_divergence/
    get_sample_and_token_indices all access `.input_ids` / `.attention_mask` as
    attributes (the normal tokenizer-output interface), while model(**inputs) still
    needs dict-style unpacking -- BatchEncoding supports both.
    """
    input_ids = torch.full((len(id_lists), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(id_lists), max_length), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        n = min(len(ids), max_length)
        input_ids[i, max_length - n:] = torch.tensor(ids[:n], dtype=torch.long)
        attention_mask[i, max_length - n:] = 1
    return BatchEncoding({'input_ids': input_ids.to(device), 'attention_mask': attention_mask.to(device)})


def run(args):
    start_time = time()
    model_name = MODEL_INFO[args.model]['model_name']
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if MODEL_INFO[args.model]['dtype'] == 'fp16' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation=MODEL_INFO[args.model]['attn_implementation'], torch_dtype=dtype,
    ).to(device)
    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > 128]
    texts = random.Random(1).sample(texts, args.n_samples)

    BATCH_SIZE = args.batch_size if args.batch_size > 0 else MODEL_INFO[args.model]['max_batch_size']

    print(f'Generating paired hotflip + token_coupled for {len(texts)} texts (this is the slow part)...')
    base_ids, hotflip_ids, coupled_ids, hotflip_texts, coupled_texts = hotflip_and_coupled_random(
        texts, args.ptb_pct, rng, tokenizer, model=model, device=device, n_candidates=args.n_candidates,
    )

    conditions = {
        'hotflip_paired': (hotflip_ids, hotflip_texts),
        'token_coupled': (coupled_ids, coupled_texts),
    }
    pct_dirs = {}
    for ptb_type, (ids_perturbed, texts_perturbed) in conditions.items():
        pct_dir = f'{args.out_root}/{args.model}/{ptb_type}/{args.ptb_pct}'
        os.makedirs(pct_dir, exist_ok=True)
        pct_dirs[ptb_type] = pct_dir
        write_manifest(pct_dir, args, ptb_type)
        write_texts_csv(pct_dir, texts, texts_perturbed, base_ids, ids_perturbed)

    seen = {}
    for ptb_type, pct_dir in pct_dirs.items():
        try:
            seen[ptb_type] = set(pd.read_csv(f'{pct_dir}/evals.csv', usecols=['sample'])['sample'])
        except Exception:
            seen[ptb_type] = set()

    pad_token_id = tokenizer.pad_token_id
    eval_hidden = MODEL_INFO[args.model]['eval_hidden_states']
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"paired pct={args.ptb_pct}"):
        batch_base_ids = base_ids[i:i + BATCH_SIZE]
        inputs = build_batch_tensors(batch_base_ids, pad_token_id, 128, device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=eval_hidden, output_attentions=eval_hidden)

        for ptb_type, (ids_perturbed, _) in conditions.items():
            batch_perturbed_ids = ids_perturbed[i:i + BATCH_SIZE]
            inputs_perturbed = build_batch_tensors(batch_perturbed_ids, pad_token_id, 128, device)
            with torch.no_grad():
                outputs_perturbed = model(**inputs_perturbed, output_hidden_states=eval_hidden,
                                           output_attentions=eval_hidden)
            res = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, i,
                             output_only=(not eval_hidden))
            res = res[~res['sample'].isin(seen[ptb_type])]
            res.to_csv(f'{pct_dirs[ptb_type]}/evals.csv', mode='a',
                       header=(i == 0 and len(seen[ptb_type]) == 0), index=False, float_format='%.6f')
            del outputs_perturbed

        del outputs

    print(f"Total time taken: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired HotFlip + position-coupled random control")
    parser.add_argument("--model", type=str, default='gpt2')
    parser.add_argument("--ptb-pct", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--out-root", type=str, default='results')
    parser.add_argument("--n-candidates", type=int, default=50)
    args = parser.parse_args()
    print('Running with', args)
    run(args)
