import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import os
import pandas as pd

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

def characterize_entropy_curve(ptb_percentages, entropy_values):
    entropy_values = np.array(entropy_values)
    
    peak_idx = np.argmax(entropy_values)
    peak_pct = ptb_percentages[peak_idx]
    peak_val = entropy_values[peak_idx]

    baseline_val = entropy_values[0]
    
    endpoint_val = entropy_values[-1]
    
    rise = peak_val - baseline_val
    fall = peak_val - endpoint_val
    net = endpoint_val - baseline_val
    
    if rise < 0.01 and abs(net) < 0.01:
        shape = 'flat'
    elif rise > 0.01 and fall < rise * 0.3:
        shape = 'monotone_increase'
    elif net < -0.01 and rise < 0.01:
        shape = 'monotone_decrease'
    elif rise > 0.01 and fall > rise * 0.5:
        shape = 'u_shaped'
    else:
        shape = 'ambiguous'
    
    return peak_pct, {
        'peak_pct': peak_pct,
        'peak_val': peak_val,
        'baseline_val': baseline_val,
        'endpoint_val': endpoint_val,
        'rise': rise,
        'fall': fall,
        'net_slope': net,
        'shape': shape
    }

def score_head(curve_stats, std_at_peak):
    if curve_stats['shape'] == 'u_shaped':
        signal = curve_stats['rise'] + curve_stats['fall']
    elif curve_stats['shape'] == 'monotone_increase':
        signal = curve_stats['rise']
    elif curve_stats['shape'] == 'monotone_decrease':
        signal = abs(curve_stats['net_slope'])
    else:
        signal = 0
    
    return signal * std_at_peak

def id_ablation_heads_entropy(args):
    ptb_type = args.ptb_type
    ptb_pct = [x*5 for x in range(1, 11)] if ptb_type != 'shuffle' else [x*5 for x in range(1, 21)]

    scores = {}

    for i in range(12):
        for h in range(12):
            entropy_vals = []
            for pct in ptb_pct:
                res = pd.read_csv(f'results/{ptb_type}/{pct}/evals.csv')
                entropy_vals.append(np.mean(res[f'attn_layer{i}_head_{h}_entropy']))
            peak_pct, curve_stats = characterize_entropy_curve(ptb_pct, entropy_vals)
            res_pk = pd.read_csv(f'results/{ptb_type}/{peak_pct}/evals.csv')
            scores[(i, h, peak_pct)] = score_head(curve_stats, np.std(res_pk[f'attn_layer{i}_head_{h}_entropy']))
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_heads = ranked[:10]
    
    return [x[0] for x in top_heads]

def id_ablation_heads_activation(args):
    ptb_type = args.ptb_type
    ptb_pct = 25 if ptb_type != 'shuffle' else 50
    sim_drop_by_layer = []

    baseline = pd.read_csv('results/char/0/evals.csv')
    perturbed = pd.read_csv(f'results/{ptb_type}/{ptb_pct}/evals.csv')

    for i in range(12):
        baseline_sim = np.mean(baseline[f'activation_cos_sim_layer_{i}'])
        perturbed_sim = np.mean(perturbed[f'activation_cos_sim_layer_{i}'])
        sim_drop_by_layer.append((i, baseline_sim - perturbed_sim))

    sim_drop_by_layer.sort(key=lambda x: x[1], reverse=True)
    target_layers = [layer for layer, drop in sim_drop_by_layer[:3]]

    return [(l, i, ptb_pct) for i in range(12) for l in target_layers]


def run_ablation(args, ablate_type, ptb_pct, l, h):
    SEQ_LEN = 128 if not args.debug else 5
    ptb_type = args.ptb_type
    
    rng = random.Random(args.seed)
    # Set up models
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        attn_implementation='eager'
    )

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


    BATCH_SIZE = 4

    from tqdm import tqdm

    os.makedirs(f'results/ablated/{ptb_type}_{ablate_type}/l={l}_h={h}_pct={ptb_pct}', exist_ok=True)

    try:
        seen = set(pd.read_csv(f'results/ablated/{ptb_type}_{ablate_type}/l={l}_h={h}_pct={ptb_pct}/evals.csv')['sample'])
    except:
        seen = set()

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={ptb_pct}"):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_texts_perturbed = texts_perturbed[i:i+BATCH_SIZE]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                        truncation=True, max_length=128, padding='max_length')
        inputs_perturbed = tokenizer(batch_texts_perturbed, return_tensors="pt",
                                    truncation=True, max_length=128, padding='max_length')
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            outputs_perturbed = model(**inputs_perturbed, output_hidden_states=True, output_attentions=True)
        
        res = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, i)

        res = res[~res['sample'].isin(seen)]
        if not args.debug:
            res.to_csv(f'results/ablated/{ptb_type}_{ablate_type}/l={l}_h={h}_pct={ptb_pct}/evals.csv', 
                                    mode='a', header=(i==0 and len(seen) == 0), index=False)
        else:
            res.to_csv('results/debug.csv', mode='a', header=(i==0 and len(seen) == 0), index=False)
        
        del outputs, outputs_perturbed
    handle.remove() 

def run(args):
    entropy_heads = id_ablation_heads_entropy(args)
    for l, h, pct in entropy_heads:
        print('entropy', pct, l, h)
        # run_ablation(args, 'entropy', pct, l, h)
    
    activation_heads = id_ablation_heads_activation(args)
    for l, h, pct in activation_heads:
        print('activation', pct, l, h)
        # run_ablation(args, 'activation', pct, l, h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle']", type=str, default='char')
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running ablation with', args)

    run(args)