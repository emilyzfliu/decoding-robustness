import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import os
import pandas as pd

from src.perturbs import perturb
from src.eval import eval_loop

def run(args):
    SEQ_LEN = 128 if not args.debug else 5
    ptb_type = args.ptb_type
    ptb_pct = args.ptb_pct
    
    rng = random.Random(args.seed)
    # Set up models
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        attn_implementation='eager'
    )

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

    os.makedirs(f'results/{ptb_type}/{ptb_pct}', exist_ok=True)

    try:
        seen = set(pd.read_csv(f'results/{ptb_type}/{ptb_pct}/sequence_evals.csv')['sample'])
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
            res.to_csv(f'results/{ptb_type}/{ptb_pct}/evals.csv', 
                                    mode='a', header=(i==0 and len(seen) == 0), index=False)
        else:
            res.to_csv('results/debug.csv', mode='a', header=(i==0 and len(seen) == 0), index=False)
        
        del outputs, outputs_perturbed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle', 'noise']", type=str, default='char')
    parser.add_argument("--ptb-pct", help="Percent of input text perturbed", type=int, default=0)
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running with', args)

    run(args)