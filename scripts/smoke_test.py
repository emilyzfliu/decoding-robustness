"""
Memory / runtime smoke test: loads each model, runs one tiny forward pass with
output_hidden_states + output_attentions (as eval_loop does), reports peak VRAM.

Usage: python smoke_test.py [--models gpt2,gpt2-xl]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_INFO, CROSS_MODEL_MODELS


def probe(model_key, batch_size=2, seq_len=128):
    info = MODEL_INFO[model_key]
    dtype = torch.float16 if info["dtype"] == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(info["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        info["model_name"], attn_implementation="eager", dtype=dtype
    ).to(device)
    model.eval()
    load_s = time.time() - t0

    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, min(tokenizer.vocab_size, 50000), (batch_size, seq_len)).to(
        device
    )
    mask = torch.ones_like(ids)
    t0 = time.time()
    with torch.no_grad():
        model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=info["eval_hidden_states"],
            output_attentions=info["eval_hidden_states"],
        )
    fwd_s = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.empty_cache()
    return load_s, fwd_s, peak_gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(CROSS_MODEL_MODELS))
    args = parser.parse_args()

    print(
        f'{"model":<16}{"dtype":<6}{"weights":>8}{"load_s":>8}{"fwd_s":>8}{"peak_GB":>9}{"VRAM%":>7}'
    )
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    for key in args.models.split(","):
        key = key.strip()
        if key not in MODEL_INFO:
            print(f"{key:<16} NOT IN MODEL_INFO")
            continue
        info = MODEL_INFO[key]
        try:
            load_s, fwd_s, peak = probe(key)
            print(
                f'{key:<16}{info["dtype"]:<6}{"~":>8}{load_s:>8.1f}{fwd_s:>8.2f}'
                f"{peak:>9.2f}{peak / total * 100:>6.0f}%"
            )
        except RuntimeError as e:
            print(f"{key:<16} RUNTIME ERROR: {str(e)[:80]}")
        except Exception as e:
            print(f"{key:<16} ERROR: {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
