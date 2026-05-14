import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random 
from src.data import load_eval_dataset

def run():
    rng = random.Random(1)
    # Set up models
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2"
    )

    # Set up dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    texts = load_eval_dataset(n_samples=10, rng=rng, perturb_pct=0)

    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=20,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode results
    outs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    for out in outs:
        print('SAMPLE:', out)


if __name__ == "__main__":
    run()