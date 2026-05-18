import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random 
import argparse
import string

from src.perturbs import perturb
from src.eval import eval_loop

def run(args):
    ptb_type = args.ptb_type
    ptb_pct = args.ptb_pct
    
    rng = random.Random(1)
    # Set up models
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2"
    )

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > 20]

    if args.debug:
        texts = texts[:2]
    
    # print(texts)

    if ptb_type == 'char':
        sub_pool = string.ascii_letters + string.digits + string.punctuation
    elif ptb_type == 'token':
        sub_pool = [
            token for token in tokenizer.get_vocab() 
            if token not in tokenizer.all_special_tokens
        ]

    texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, sub_pool)

    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs_perturbed = tokenizer(texts_perturbed, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(
            **inputs, 
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_logits=True
        )

        outputs_perturbed = model(
            **inputs_perturbed, 
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_logits=True
        )
    
    res = eval_loop(inputs, outputs, inputs_perturbed, outputs_perturbed, tokenizer, model)

    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--ptb-type", help="Perturbation type: ['char', 'token', 'shuffle', 'noise']", type=str, default='char')
    parser.add_argument("--ptb-pct", help="Percent of input text perturbed", type=int, default=0)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    run(args)