import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import string
import os
import pandas as pd

from src.perturbs import perturb
from src.eval import eval_loop

def run(args):
    SEQ_LEN = 128 if not args.debug else 5
    ptb_type = args.ptb_type
    ptb_pct = args.ptb_pct
    
    rng = random.Random(1)
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

    if args.debug:
        texts = [
            'Lorem ipsum dolor sit amet',
            'Hello world! Hello universe?'
        ]
    
    # print(texts)

    if ptb_type == 'char':
        sub_pool = string.ascii_letters + string.digits + string.punctuation
    elif ptb_type == 'token':
        sub_pool = [
            token for token in tokenizer.get_vocab() 
            if token not in tokenizer.all_special_tokens
        ]

    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, sub_pool)

    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True,
        max_length=SEQ_LEN,
        padding='max_length' 
    )
    inputs_perturbed = tokenizer(
        texts_perturbed, 
        return_tensors="pt", 
        truncation=True,
        max_length=SEQ_LEN,
        padding='max_length' 
    )

    outputs = model(
        **inputs, 
        output_hidden_states=True, 
        output_attentions=True
    )

    outputs_perturbed = model(
        **inputs_perturbed,
        output_hidden_states=True, 
        output_attentions=True
    )
    
    res_seq, res_tok = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, model)

    os.makedirs(f'results/{ptb_type}/{ptb_pct}', exist_ok=True)

    res_seq = pd.DataFrame(res_seq)
    res_tok = pd.DataFrame(res_tok)

    res_seq.to_csv(f'results/{ptb_type}/{ptb_pct}/sequence_evals.csv', index=False)
    res_tok.to_csv(f'results/{ptb_type}/{ptb_pct}/token_evals.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle', 'noise']", type=str, default='char')
    parser.add_argument("--ptb-pct", help="Percent of input text perturbed", type=int, default=0)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    print('Running with', args)

    run(args)