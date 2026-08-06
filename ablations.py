import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import os
import pandas as pd
from tqdm import tqdm
from time import time

from src.perturbs import perturb
from src.eval import eval_loop
from config import MODEL_INFO
    
def ablate_head(model, layer_idx, head_idx):
    def hook(module, input, output):
        head_size = 64
        start = head_idx * head_size
        end = start + head_size
        output[0][:, :, start:end] = 0
        return output
    
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(hook)
    return handle


def entropy_slope_by_head(ptb_type, percentages, model='gpt2'):
    # shape: (n_pcts, n_layers, n_heads)
    entropy_by_pct = []
    
    for pct in percentages:
        seq = pd.read_csv(f'results/{model}/{ptb_type}/{pct}/evals.csv')
        
        layer_head_means = []
        for i in range(12):
            head_means = []
            for h in range(12):
                col = f'attn_layer{i}_head_{h}_entropy_norm'
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

    slopes = entropy_slope_by_head(ptb_type, ptb_pct, model=args.model)

    for i in range(12):
        for h in range(12):
            score = abs(slopes[i, h])
            scores[(i, h, ptb_pct[-1], 'diffuse' if slopes[i, h] > 0 else 'sink')] = score
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_heads = ranked[:5]
    
    return [x[0] for x in top_heads]

def run_ablation(args, ablate_type, ptb_pct, l, h):
    start_time = time()
    SEQ_LEN = 128 if not args.debug else 5

    model_name = MODEL_INFO[args.model]['model_name']
        
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tagline = args.output_tag if hasattr(args, 'output_tag') else ''
    # Set up models

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      attn_implementation=MODEL_INFO[args.model]['attn_implementation']
    ).to(device)

    model.eval()

    handle = ablate_head(model, layer_idx=l, head_idx=h)

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
    
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else (128 if torch.cuda.is_available() else 4)
    
    ptb_type = args.ptb_type

    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer)

    os.makedirs(f'results_{tagline}/{args.model}/{ptb_type}/{ptb_pct}', exist_ok=True)

    try:
        seen = set(pd.read_csv(f'results_{tagline}/{args.model}/{ptb_type}/{ptb_pct}/evals.csv')['sample'])
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
            res.to_csv(f'results_{tagline}/{args.model}/{ptb_type}/{ptb_pct}/evals.csv', 
                                    mode='a', header=(i==0 and len(seen) == 0), index=False)
        else:
            res.to_csv(f'results_{tagline}/{args.model}/debug.csv', mode='a', header=(i==0 and len(seen) == 0), index=False)
        
        del outputs, outputs_perturbed
    handle.remove() 
    print(f"Total time taken: {time() - start_time:.2f} seconds")

def run(args):
    # entropy_heads = id_ablation_heads_entropy(args)
    # for l, h, pct, classif in entropy_heads:

    for l in range(12):
        for h in range(12):
            classif = ''
            run_ablation(args, classif, 5, l, h)
            run_ablation(args, classif, 30, l, h)

            if args.ptb_type == 'char':
                # perturbed baseline
                run_ablation(args, classif, 0, l, h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--model", help="Model name (must be in config.MODEL_INFO)", type=str, default='gpt2')
    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle']", type=str, default='char')
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--output-tag", help="Tag for output directory", type=str, default='')

    args = parser.parse_args()

    print('Running ablation with', args)

    run(args)