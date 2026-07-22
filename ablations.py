import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import os
import pandas as pd

import numpy as np
from scipy import stats

from src.perturbs import perturb
from src.eval import eval_loop
    
def ablate_head(model, layer_idx, head_idx):
    def hook(module, input, output):
        head_size = 64
        start = head_idx * head_size
        end = start + head_size
        output[0][:, :, start:end] = 0
        return output
    
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook)
    return handle


def entropy_slope_by_head(ptb_type, percentages):
    # shape: (n_pcts, n_layers, n_heads)
    entropy_by_pct = []
    
    for pct in percentages:
        seq = pd.read_csv(f'results/{ptb_type}/{pct}/evals.csv')
        
        layer_head_means = []
        for i in range(12):
            head_means = []
            for h in range(12):
                col = f'attn_layer{i}_head_{h}_entropy'
                head_means.append(np.mean(seq[col]))
            layer_head_means.append(head_means)
        entropy_by_pct.append(layer_head_means)
    
    entropy_by_pct = np.array(entropy_by_pct)
    
    slopes = np.zeros((12, 12))
    for i in range(12):
        for h in range(12):
            slope, _, _, _, _ = stats.linregress(percentages, entropy_by_pct[:, i, h])
            slopes[i, h] = slope
    
    return slopes

def id_ablation_heads_entropy(args):
    ptb_type = args.ptb_type
    ptb_pct = [x*5 for x in range(1, 11)] if ptb_type != 'shuffle' else [x*5 for x in range(1, 21)]

    scores = {}

    slopes = entropy_slope_by_head(ptb_type, ptb_pct)

    for i in range(12):
        for h in range(12):
            score = abs(slopes[i, h])
            scores[(i, h, ptb_pct[-1], 'diffuse' if slopes[i, h] > 0 else 'sink')] = score
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_heads = ranked[:5]
    
    return [x[0] for x in top_heads]

def run_ablation(args, ablate_type, ptb_pct, l, h):
    SEQ_LEN = 128 if not args.debug else 5
    ptb_type = args.ptb_type
    
    rng = random.Random(args.seed)
    # Set up models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        attn_implementation='eager'
    ).to(device)
    model.eval()

    handle = ablate_head(model, layer_idx=l, head_idx=h)

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]

    rng_data = random.Random(1)

    texts = rng_data.sample(texts, 100)

    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet',
            'Hello world! Hello universe?'
        ]
    

    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer)


    BATCH_SIZE = 16

    from tqdm import tqdm

    os.makedirs(f'results/ablated/{ptb_type}/l={l}_h={h}_pct={ptb_pct}_{ablate_type}', exist_ok=True)

    try:
        seen = set(pd.read_csv(f'results/ablated/{ptb_type}/l={l}_h={h}_pct={ptb_pct}_{ablate_type}/evals.csv')['sample'])
    except:
        seen = set()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={ptb_pct}"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_texts_perturbed = texts_perturbed[i:i+BATCH_SIZE]
        
        inputs = tokenizer(batch_texts, return_tensors="pt",
                        truncation=True, max_length=128, padding='max_length').to(device)
        inputs_perturbed = tokenizer(batch_texts_perturbed, return_tensors="pt",
                                    truncation=True, max_length=128, padding='max_length').to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            outputs_perturbed = model(**inputs_perturbed, output_hidden_states=True, output_attentions=True)
        
        res = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, i, output_only=True)

        res = res[~res['sample'].isin(seen)]
        if not args.debug:
            res.to_csv(f'results/ablated/{ptb_type}/l={l}_h={h}_pct={ptb_pct}_{ablate_type}/evals.csv', 
                                    mode='a', header=(i==0 and len(seen) == 0), index=False)
        else:
            res.to_csv('results/debug.csv', mode='a', header=(i==0 and len(seen) == 0), index=False)
        
        del outputs, outputs_perturbed
    handle.remove() 

def run(args):
    entropy_heads = id_ablation_heads_entropy(args)
    for l, h, pct, classif in entropy_heads:
        run_ablation(args, classif, pct, l, h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle']", type=str, default='char')
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running ablation with', args)

    run(args)