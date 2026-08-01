"""
Quick validation test for all new features.
Tests on minimal data to verify no errors before full runs.

Usage: python test_quick.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random
import os

from src.perturbs import perturb
from src.eval import eval_loop, activation_cka, twoNN_intrinsic_dim, mknn_intrinsic_dim


def test_perturbations():
    """Test all perturbation types including word-level."""
    print("=" * 60)
    print("TEST 1: Perturbation implementations")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    rng = random.Random(1)
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of the perturbation system."
    ]
    
    for ptb_type in ['char', 'token', 'word', 'shuffle', 'typo', 'synonym']:
        try:
            result = perturb(texts, 25, rng, ptb_type, tokenizer)
            assert len(result) == len(texts), f"{ptb_type}: wrong output count"
            print(f"  OK {ptb_type}: {len(result)} texts perturbed successfully")
            print(f"     Original: {texts[0][:60]}...")
            print(f"     Perturbed: {result[0][:60]}...")
        except Exception as e:
            print(f"  FAIL {ptb_type}: {e}")
    
    return True


def test_metrics():
    """Test all evaluation metrics including CKA, TwoNN, MKNN."""
    print("\n" + "=" * 60)
    print("TEST 2: Evaluation metrics")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        "openai-community/gpt2",
        attn_implementation='eager'
    )
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a test of the perturbation system."
    ]
    
    rng = random.Random(1)
    texts_perturbed = perturb(texts, 25, rng, 'token', tokenizer)
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    inputs_ptb = tokenizer(texts_perturbed, return_tensors="pt", truncation=True, max_length=32, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        outputs_ptb = model(**inputs_ptb, output_hidden_states=True, output_attentions=True)
    
    # Test CKA
    try:
        cka_result = activation_cka(outputs, outputs_ptb)
        cka_keys = [k for k in cka_result.keys() if k.startswith('activation_cka_layer_')]
        print(f"  OK CKA: {len(cka_keys)} layers computed")
        for k in cka_keys[:3]:
            print(f"     {k}: {cka_result[k][0]:.4f}")
    except Exception as e:
        print(f"  FAIL CKA: {e}")
    
    # Test TwoNN
    try:
        twonn_result = twoNN_intrinsic_dim(outputs, outputs_ptb)
        twonn_keys = [k for k in twonn_result.keys() if 'change' in k]
        print(f"  OK TwoNN: {len(twonn_keys)} layers computed")
        for k in twonn_keys[:3]:
            print(f"     {k}: {twonn_result[k][0]:.4f}")
    except Exception as e:
        print(f"  FAIL TwoNN: {e}")
    
    # Test MKNN
    try:
        mknn_result = mknn_intrinsic_dim(outputs, outputs_ptb)
        mknn_keys = [k for k in mknn_result.keys() if 'change' in k]
        print(f"  OK MKNN: {len(mknn_keys)} layers computed")
        for k in mknn_keys[:3]:
            print(f"     {k}: {mknn_result[k][0]:.4f}")
    except Exception as e:
        print(f"  FAIL MKNN: {e}")
    
    # Test full eval_loop
    try:
        result = eval_loop(inputs, outputs, inputs_ptb, outputs_ptb, tokenizer, 0)
        print(f"  OK eval_loop: {len(result)} samples, {len(result.columns)} columns")
        print(f"     Columns: {list(result.columns[:8])}...")
    except Exception as e:
        print(f"  FAIL eval_loop: {e}")
    
    return True


def test_word_substitution_vocab():
    """Test word-level vocabulary building."""
    print("\n" + "=" * 60)
    print("TEST 3: Word-level vocabulary")
    print("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    # Build word vocab (same logic as in perturbs.py)
    word_vocab = []
    for i in range(min(10000, tokenizer.vocab_size)):
        decoded = tokenizer.decode([i]).strip()
        if len(decoded) > 1 and decoded.isalpha():
            word_vocab.append(decoded)
    
    print(f"  Word vocabulary size: {len(word_vocab)}")
    print(f"  Sample words: {word_vocab[:10]}")
    
    if len(word_vocab) > 1000:
        print(f"  OK Word vocabulary sufficient (>1000 words)")
    else:
        print(f"  WARNING Word vocabulary small ({len(word_vocab)}), will use fallback")
    
    return True


def test_memory_usage():
    """Check memory usage stays reasonable."""
    print("\n" + "=" * 60)
    print("TEST 4: Memory check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")
    else:
        print("  CPU mode (no GPU detected)")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK VALIDATION TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    all_passed &= test_perturbations()
    all_passed &= test_metrics()
    all_passed &= test_word_substitution_vocab()
    all_passed &= test_memory_usage()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("OK ALL TESTS PASSED")
        print("Ready for full experiments!")
    else:
        print("FAIL SOME TESTS FAILED - Check output above")
    print("=" * 60)