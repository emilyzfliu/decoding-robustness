"""
k-sensitivity diagnostic for the representation metric.

Drops the top-k highest-variance (massive-activation) dims per layer, then
compares stripped CKA vs stripped cosine. When the massive-activation confound
is fully removed the two metrics agree; the smallest k where they converge is
the right k for this model. (k=5, tuned on GPT-2, is too small for Llama.)

Run with the same env as the sweep (HF_HOME + your HF login/token):
  python llama_k_diagnostic.py --model meta-llama/Llama-3.1-8B --ptb-type char --ptb-pct 25 --n 40
"""
import argparse, random
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.perturbs import perturb
from src.eval import _linear_cka, _drop_top_var_dims

KS = [0, 5, 10, 20, 50, 100, 200, 500]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ptb-type", default="char")
    ap.add_argument("--ptb-pct", type=int, default=25)
    ap.add_argument("--n", type=int, default=40, help="number of passages to use")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(device).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = [x for x in ds["test"]["text"] if len(x.split()) > 128]
    texts = random.Random(1).sample(texts, 100)[:args.n]           # same sampling as main.py
    pert = perturb(texts, args.ptb_pct, random.Random(args.seed), args.ptb_type, tok)

    cka_sum, cos_sum = defaultdict(float), defaultdict(float)
    nseen, nL = 0, None
    for i in range(0, len(texts), args.batch_size):
        xb = tok(texts[i:i+args.batch_size], return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
        xp = tok(pert[i:i+args.batch_size],  return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
        with torch.no_grad():
            hb = model(**xb, output_hidden_states=True).hidden_states
            hp = model(**xp, output_hidden_states=True).hidden_states
        nL = len(hb); bs = xb["input_ids"].shape[0]; nseen += bs
        for L in range(nL):
            A = hb[L][:, :-1, :].float(); B = hp[L][:, :-1, :].float()
            for b in range(bs):
                X, Y = A[b], B[b]
                for k in KS:
                    Xs, Ys = _drop_top_var_dims(X, Y, k)
                    cka_sum[(L, k)] += _linear_cka(Xs, Ys)
                    cos_sum[(L, k)] += torch.cosine_similarity(Xs, Ys, dim=-1).mean().item()

    report = [0, 1, 2, 4, nL // 4, nL // 2, 3 * nL // 4, nL - 1]
    print(f"\nmodel={args.model}  {args.ptb_type}@{args.ptb_pct}%  n={nseen}  layers={nL}")
    print("Right k = smallest k where CKA ~ cos (|gap| small).\n")
    for L in sorted(set(report)):
        print(f"--- layer {L} ---   {'k':>5} {'CKA':>7} {'cos':>7} {'gap':>7}")
        conv = None
        for k in KS:
            cka, cos = cka_sum[(L, k)] / nseen, cos_sum[(L, k)] / nseen
            print(f"{'':>15} {k:>5} {cka:>7.3f} {cos:>7.3f} {cka - cos:>+7.3f}")
            if conv is None and abs(cka - cos) < 0.05:
                conv = k
        print(f"    -> converges (|gap|<0.05) at k={conv}\n")

if __name__ == "__main__":
    main()
