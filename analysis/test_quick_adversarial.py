"""
Quick validation test for the adversarial/injection perturbations.
Tests on minimal data to verify no errors before full runs.

Usage: python test_quick_adversarial.py
"""
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.adversarial_perturbs import context_insertion, question_perturbation
from src.hotflip import hotflip_attack

TEXTS = [
    "The quick brown fox jumps over the lazy dog in the year 1995 near Paris.",
    "Hello world! This is a test of the perturbation system written by John Smith.",
]


def _load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation='eager')
    except TypeError:
        # Older transformers (this repo's installed 4.24.0) doesn't accept attn_implementation.
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.eval()
    return model, tokenizer


def test_context_insertion(model, tokenizer):
    print("=" * 60)
    print("TEST 1: context_insertion")
    print("=" * 60)

    rng = random.Random(1)
    device = torch.device("cpu")

    for condition in ['topic_shift', 'misleading_claim', 'adversarial']:
        try:
            result = context_insertion(
                TEXTS, condition, rng, tokenizer, model=model, device=device,
                distractor_pool=TEXTS, insertion_len=4, n_iters=3, max_length=32,
            )
            assert len(result) == len(TEXTS), f"{condition}: wrong output count"
            changed = sum(1 for a, b in zip(TEXTS, result) if a != b)
            print(f"  OK {condition}: {changed}/{len(TEXTS)} texts changed")
            print(f"     Original:  {TEXTS[0][:70]}...")
            print(f"     Perturbed: {result[0][:70]}...")
        except Exception as e:
            print(f"  FAIL {condition}: {e}")
    return True


def test_question_perturbation(model, tokenizer):
    print("\n" + "=" * 60)
    print("TEST 2: question_perturbation")
    print("=" * 60)

    rng = random.Random(1)
    device = torch.device("cpu")
    contexts = [' '.join(t.split()[:-1]) for t in TEXTS]

    for condition in ['synonym', 'reorder', 'negation_paraphrase', 'adversarial_swap']:
        try:
            result = question_perturbation(
                contexts, condition, rng, tokenizer, model=model, device=device, max_length=32,
            )
            assert len(result) == len(contexts), f"{condition}: wrong output count"
            changed = sum(1 for a, b in zip(contexts, result) if a != b)
            print(f"  OK {condition}: {changed}/{len(contexts)} texts changed")
            print(f"     Original:  {contexts[0][:70]}...")
            print(f"     Perturbed: {result[0][:70]}...")
        except RuntimeError as e:
            # Expected if `python -m nltk.downloader wordnet` hasn't been run yet.
            print(f"  SKIP {condition}: {e}")
        except Exception as e:
            print(f"  FAIL {condition}: {e}")
    return True


def test_hotflip_attack(model, tokenizer):
    print("\n" + "=" * 60)
    print("TEST 3: hotflip_attack (loss should increase)")
    print("=" * 60)

    rng = random.Random(1)
    device = torch.device("cpu")

    ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"][0]
    attention_mask = torch.ones_like(ids)
    positions = list(range(len(ids)))

    def seq_loss(input_ids):
        with torch.no_grad():
            out = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        labels = input_ids.clone()
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[1:].unsqueeze(0)
        return torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
        ).item()

    try:
        before = seq_loss(ids)
        attacked = hotflip_attack(model, tokenizer, ids, attention_mask, positions, device,
                                   rng, n_candidates=20, n_iters=5)
        after = seq_loss(attacked)
        print(f"  Loss before: {before:.4f}, after: {after:.4f}")
        if after >= before:
            print("  OK hotflip_attack increased (or held) the sequence loss")
        else:
            print("  FAIL hotflip_attack decreased the sequence loss")
    except Exception as e:
        print(f"  FAIL hotflip_attack: {e}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ADVERSARIAL PERTURBATIONS - QUICK VALIDATION")
    print("=" * 60)

    model, tokenizer = _load_model_and_tokenizer()

    test_context_insertion(model, tokenizer)
    test_question_perturbation(model, tokenizer)
    test_hotflip_attack(model, tokenizer)

    print("\n" + "=" * 60)
    print("DONE - check FAIL lines above (SKIP is expected if wordnet isn't downloaded yet)")
    print("=" * 60)
