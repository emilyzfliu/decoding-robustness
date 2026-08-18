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


def run(args):
    start_time = time()
    SEQ_LEN = 128 if not args.debug else 5

    model_name = MODEL_INFO[args.model]["model_name"]

    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up models

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = (
        torch.float16 if MODEL_INFO[args.model]["dtype"] == "fp16" else torch.float32
    )
    if device.type != "cuda":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=MODEL_INFO[args.model]["attn_implementation"],
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds["test"]["text"])
    texts = [x for x in texts if len(x.split()) > SEQ_LEN]
    if args.n_samples > 0 and not args.debug:
        texts = random.Random(1).sample(texts, args.n_samples)
    if args.debug:
        texts = ["Lorem ipsum dolor sit amet", "Hello world! Hello universe?"]

    BATCH_SIZE = (
        args.batch_size
        if args.batch_size > 0
        else MODEL_INFO[args.model]["max_batch_size"]
    )

    ptb_type = args.ptb_type

    if args.ptb_pct != -1:
        ptb_pcts = [args.ptb_pct]
    else:
        if ptb_type == "char":
            ptb_pcts = [x * 5 for x in range(0, 11)]  # baseline [hacky]
        else:
            ptb_pcts = [x * 5 for x in range(1, 11)]

    for ptb_pct in ptb_pcts:

        texts_perturbed = perturb(texts, ptb_pct, rng, ptb_type, tokenizer, model=model)

        os.makedirs(f"{args.out_root}/{args.model}/{ptb_type}/{ptb_pct}", exist_ok=True)

        try:
            seen = set(
                pd.read_csv(
                    f"{args.out_root}/{args.model}/{ptb_type}/{ptb_pct}/evals.csv",
                    usecols=["sample"],
                )["sample"]
            )
        except:
            seen = set()

        for i in tqdm(
            range(0, len(texts), BATCH_SIZE), desc=f"{ptb_type} pct={ptb_pct}"
        ):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_texts_perturbed = texts_perturbed[i : i + BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length",
            ).to(device)
            inputs_perturbed = tokenizer(
                batch_texts_perturbed,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length",
            ).to(device)

            eval_hidden = MODEL_INFO[args.model]["eval_hidden_states"]
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=eval_hidden,
                    output_attentions=eval_hidden,
                )
                outputs_perturbed = model(
                    **inputs_perturbed,
                    output_hidden_states=eval_hidden,
                    output_attentions=eval_hidden,
                )

            res = eval_loop(
                inputs,
                outputs,
                inputs_perturbed,
                outputs_perturbed,
                tokenizer,
                i,
                output_only=(not eval_hidden),
            )

            res = res[~res["sample"].isin(seen)]
            if not args.debug:
                res.to_csv(
                    f"{args.out_root}/{args.model}/{ptb_type}/{ptb_pct}/evals.csv",
                    mode="a",
                    header=(i == 0 and len(seen) == 0),
                    index=False,
                    float_format="%.6f",
                )
            else:
                res.to_csv(
                    f"{args.out_root}/{args.model}/debug.csv",
                    mode="a",
                    header=(i == 0 and len(seen) == 0),
                    index=False,
                    float_format="%.6f",
                )

            del outputs, outputs_perturbed
    print(f"Total time taken: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params: perturb_type, perturb_pct")

    parser.add_argument("--model", help="Model name", type=str, default="gpt2")
    parser.add_argument(
        "--ptb-type",
        help="Perturbation type: ['char', 'token', 'shuffle']",
        type=str,
        default="char",
    )
    parser.add_argument(
        "--ptb-pct", help="Percent of input text perturbed", type=int, default=-1
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parser.add_argument(
        "--n-samples",
        help="Number of sequences to sample (-1 = use all)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size (<=0 = per-model max_batch_size)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--out-root",
        help="Root directory for results (default: results)",
        type=str,
        default="results",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("Running with", args)

    run(args)
