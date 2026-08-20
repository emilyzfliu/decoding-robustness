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
from time import time

from functools import reduce


from src.perturbs import perturb
from config import MODEL_INFO


def attention_entropy(attentions):
    """
    Aggregated per-(layer, head) entropy summary. One row per head.
    """
    ret = {
       'head': [],
       'layer': [],
       'ent_norm_mean': [],
       'ent_norm_median': []
    }
    for layer_idx in range(len(attentions)):
        layer_att = attentions[layer_idx].float()
        _, nh, seq_len, _ = layer_att.shape
        head_att = layer_att[:, :, :-1, :]
        mask = head_att > 0
        safe_att = head_att.clamp(min=1e-9)
        ent = -torch.sum(mask * head_att * torch.log(safe_att), dim=-1)

        positions = torch.arange(1, seq_len, device=ent.device, dtype=torch.float32)
        max_ent_per_pos = torch.log(positions)
        ent_norm = ent / max_ent_per_pos.unsqueeze(0).unsqueeze(0)

        ent_norm_mean = torch.nanmean(ent_norm, dim=[0, 2])

        ent_norm_flat = ent_norm.permute(1, 0, 2).reshape(nh, -1) 
        ent_norm_median = torch.nanmedian(ent_norm_flat, dim=1).values

        for head_idx in range(nh):
            ret['layer'].append(layer_idx)
            ret['head'].append(head_idx)
            ret['ent_norm_mean'].append(ent_norm_mean[head_idx].item())
            ret['ent_norm_median'].append(ent_norm_median[head_idx].item())

    return pd.DataFrame(ret)

def previous_token_score(outputs_attn_pattern):
    """
    outputs_attn_pattern: tuple of size (nl,) each element is a tensor of shape [batch, nh, seq_len, seq_len]
                  (dest_pos, src_pos) — attn_pattern[b, i, j] = attention
                  from position i to position j
    """

    ret = {
        'layer': [],
        'head': [],
        'prev_tok_score': []
    }

    for i in range(len(outputs_attn_pattern)):
        att = outputs_attn_pattern[i]
        _, nh, _, _ = att.shape
        for h in range(nh):
            attn_pattern = att[:, h, :, :]
            offset_diag = torch.diagonal(attn_pattern.nanmean(dim=[0]), offset=-1)
            ret['layer'].append(i)
            ret['head'].append(h)
            ret['prev_tok_score'].append(torch.nanmean(offset_diag))
    return pd.DataFrame(ret)


def generate_repeated_tokens(seq_len, batch_size, device, vocab_size, seed=42):
    # Samples batch_size, seq_len] random token ids, then
    # concatenates the sequence with itself along the seq dim so the
    # second half exactly repeats the first
    
    g = torch.Generator(device=device).manual_seed(seed)
    rand_tokens = torch.randint(vocab_size, (batch_size, seq_len), generator=g, device=device)

    return torch.cat([rand_tokens, rand_tokens], dim=1)

def duplicate_and_induction_scores(outputs_attn_pattern, seq_len):
    """
    outputs_attn_pattern: tuple of size (nl,) each element is a tensor of shape [batch, nh, 2*seq_len, 2*seq_len]
    """

    ret = {
        'layer': [],
        'head': [],
        'duplicate_score': [],
        'induction_score': [],
    }

    for layer_idx in range(len(outputs_attn_pattern)):
        att = outputs_attn_pattern[layer_idx]
        _, nh, _, _ = att.shape

        for head_idx in range(nh):
            attn_pattern = att[:, head_idx, :, :]

            ret['layer'].append(layer_idx)
            ret['head'].append(head_idx)
            ret['duplicate_score'].append(torch.mean(torch.diagonal(attn_pattern.nanmean(dim=[0]), offset=-seq_len)))
            ret['induction_score'].append(torch.mean(torch.diagonal(attn_pattern.nanmean(dim=[0]), offset=-(seq_len-1))))
    return pd.DataFrame(ret)


def get_head_ov_weights(model, layer_idx, head_idx):
    d_model = model.config.n_embd
    n_heads = model.config.n_head
    d_head = d_model // n_heads

    block = model.transformer.h[layer_idx].attn

    c_attn_weight = block.c_attn.weight # [d_model, 3*d_model] for Q, K, V weights
    W_V_full = c_attn_weight[:, 2 * d_model : 3 * d_model] # [d_model, d_model], isolate V
    W_V_head = W_V_full[:, head_idx * d_head : (head_idx + 1) * d_head]  # [d_model, d_head] just take the specific head

    c_proj_weight = block.c_proj.weight # [d_model, d_model]
    W_O_head = c_proj_weight[head_idx * d_head : (head_idx + 1) * d_head, :]  # [d_head, d_model]

    return W_V_head, W_O_head

def copying_score(model, token_ids, ln_scale=None, k=5):

    W_E = model.transformer.wte.weight   # [vocab_size, d_model]
    W_U = model.lm_head.weight.T          # [d_model, vocab_size]


    ret = {
        'layer': [],
        'head': [],
        'copying_score': [],
    }


    for layer_idx in range(model.config.n_layer):
        for head_idx in range(model.config.n_head):
            W_V_head, W_O_head = get_head_ov_weights(model, layer_idx, head_idx)
            
            x = W_E[token_ids]

            out = x @ W_V_head @ W_O_head

            if ln_scale is not None:
                out = out * ln_scale

            logits = out @ W_U

            
            values, _ = torch.topk(logits, k=k, dim=1)
            slices = logits[torch.arange(len(token_ids)), token_ids]

            score = torch.mean(slices >= values[:, :-1])
            ret['layer'].append(layer_idx)
            ret['head'].append(head_idx)
            ret['copying_score'].append(score)
    return pd.DataFrame(ret)



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
      dtype=dtype
    ).to(device)

    model.eval()

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]

    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet',
            'Hello world! Hello universe?'
        ]
    
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else MODEL_INFO[args.model]['max_batch_size']

    ptb_type = args.ptb_type

    ptb_pct = args.ptb_pct # do NOT sweep a full range here.
        
    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer, model=model)

    out_dir = f'{args.out_root}/{args.model}/{ptb_type}/{ptb_pct}'

    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={ptb_pct}"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_texts_perturbed = texts_perturbed[i:i+BATCH_SIZE]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                        truncation=True, max_length=128, padding='max_length').to(device)
        inputs_perturbed = tokenizer(batch_texts_perturbed, return_tensors="pt",
                                    truncation=True, max_length=128, padding='max_length').to(device)

        
        rand_inputs = generate_repeated_tokens(SEQ_LEN, BATCH_SIZE, device, tokenizer.vocab_size)
        
        eval_hidden = MODEL_INFO[args.model]['eval_hidden_states']
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=eval_hidden)
            outputs_perturbed = model(**inputs_perturbed, output_attentions=eval_hidden)

        entropy_og = attention_entropy(outputs.attentions)
        entropy_ptb = attention_entropy(outputs_perturbed.attentions)
        entropy = pd.merge(entropy_og, entropy_ptb, on=['layer', 'head'], how='inner', suffixes=('_base', '_ptb'))
        entropy['ent_norm_mean_delta'] = entropy['ent_norm_mean_base'] - entropy['ent_norm_mean_ptb']
        entropy['ent_norm_median_delta'] = entropy['ent_norm_median_base'] - entropy['ent_norm_median_ptb']


        pts_og = previous_token_score(outputs.attentions)
        pts_ptb =  previous_token_score(outputs_perturbed.attentions)
        pts = pd.merge(pts_og, pts_ptb, on=['layer', 'head'], how='inner', suffixes=('_base', '_ptb'))
        pts['prev_tok_score_delta'] = pts['prev_tok_score_base'] - pts['prev_tok_score_ptb']


        rand_outputs =  model(rand_inputs, output_attentions=True)

        dup_induction = duplicate_and_induction_scores(rand_outputs.attentions, SEQ_LEN)

        copying = copying_score(model, rand_inputs[0, :SEQ_LEN])

        dfs = [entropy, pts, dup_induction, copying]

        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['layer', 'head'], how='inner'), dfs)

        df_merged.to_csv(f'{out_dir}/attention_results.csv')
        
        del outputs, outputs_perturbed, rand_outputs
    print(f"Total time taken: {time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--model", help="Model name", type=str, default='gpt2')
    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle']", type=str, default='char')
    parser.add_argument("--ptb-pct", help="Percent of input text perturbed", type=int, default=-1)
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--batch-size", help="Batch size (<=0 = per-model max_batch_size)", type=int, default=0)
    parser.add_argument("--out-root", help="Root directory for results (default: results)", type=str, default='results')
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running with', args)

    run(args)