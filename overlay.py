"""
Cross-family overlay: stripped CKA & (CKA - cosine) gap vs RELATIVE depth, for
absolute-position (GPT-2, OPT) vs RoPE (Llama, Qwen) models.

The story it visualizes (char@25, fixed stripping budget k):
  - absolute models strip clean in early layers (gap ~0), lose it with depth;
  - RoPE models keep CKA pinned high (~0.8 plateau) with cosine near 0 at ALL
    depths -- an unstrippable, distributed positional confound.

Recomputes per-layer stripped CKA/cosine at k for n passages, caches arrays to
figures/overlay/overlay_data.npz, then plots. Re-tune the figure without GPU via
--replot (loads the cache).

Run with sweep env (HF_HOME + HF login for gated Llama). NOTE: GPT-2 may need a
one-time download -- do NOT set HF_HUB_OFFLINE=1 on the first run.
  python overlay.py                 # compute all four + plot
  python overlay.py --replot        # redraw from cache only
"""
import os, argparse, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (hf id, display label, dtype, family)  -- absolute first, then RoPE
MODELS = [
    ("openai-community/gpt2",   "GPT-2 (124M, abs)",   "float32",  "abs"),
    ("facebook/opt-6.7b",       "OPT-6.7B (abs)",      "bfloat16", "abs"),
    ("meta-llama/Llama-3.1-8B", "Llama-3.1-8B (RoPE)", "bfloat16", "rope"),
    ("Qwen/Qwen2.5-7B",         "Qwen2.5-7B (RoPE)",   "bfloat16", "rope"),
]
COLORS = {
    "openai-community/gpt2":   "#999999",
    "facebook/opt-6.7b":       "#d6604d",
    "meta-llama/Llama-3.1-8B": "#4393c3",
    "Qwen/Qwen2.5-7B":         "#2166ac",
}
STYLE = {"abs": "-", "rope": "--"}

OUTDIR = "figures/overlay"
CACHE = f"{OUTDIR}/overlay_data.npz"
os.makedirs(OUTDIR, exist_ok=True)

def compute_model(model_id, dtype, k, n, ptb_type, ptb_pct, batch_size, seed):
    """Return (nL, cka[nL], cos[nL]) of per-layer stripped metrics, averaged over n passages."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from src.perturbs import perturb
    from src.eval import _linear_cka, _drop_top_var_dims

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), attn_implementation="sdpa"
        ).to(device).eval()
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), attn_implementation="eager"
        ).to(device).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = [x for x in ds["test"]["text"] if len(x.split()) > 128]
    texts = random.Random(1).sample(texts, 100)[:n]           # same sampling as main.py
    pert = perturb(texts, ptb_pct, random.Random(seed), ptb_type, tok)

    cka_sum = cos_sum = None
    nseen = 0
    for i in range(0, len(texts), batch_size):
        xb = tok(texts[i:i+batch_size], return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
        xp = tok(pert[i:i+batch_size],  return_tensors="pt", truncation=True, max_length=128, padding="max_length").to(device)
        with torch.no_grad():
            hb = model(**xb, output_hidden_states=True).hidden_states
            hp = model(**xp, output_hidden_states=True).hidden_states
        nL = len(hb); bs = xb["input_ids"].shape[0]; nseen += bs
        if cka_sum is None:
            cka_sum, cos_sum = np.zeros(nL), np.zeros(nL)
        for L in range(nL):
            A = hb[L][:, :-1, :].float(); B = hp[L][:, :-1, :].float()
            for b in range(bs):
                Xs, Ys = _drop_top_var_dims(A[b], B[b], k)
                cka_sum[L] += _linear_cka(Xs, Ys)
                cos_sum[L] += torch.cosine_similarity(Xs, Ys, dim=-1).mean().item()

    del model
    torch.cuda.empty_cache()
    return len(cka_sum), cka_sum / nseen, cos_sum / nseen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="stripping budget (dims dropped per layer)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--ptb-type", default="char")
    ap.add_argument("--ptb-pct", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--replot", action="store_true", help="load cached arrays, skip GPU")
    args = ap.parse_args()

    data = {}  # model_id -> (depth[nL], cka[nL], cos[nL])
    if args.replot:
        z = np.load(CACHE, allow_pickle=True)
        data = z["data"].item()
        meta = z["meta"].item()
        print(f"replot from {CACHE} (k={meta['k']}, {meta['ptb_type']}@{meta['ptb_pct']}%, n={meta['n']})")
    else:
        for mid, label, dtype, fam in MODELS:
            print(f"computing {label} ...", flush=True)
            nL, cka, cos = compute_model(mid, dtype, args.k, args.n, args.ptb_type,
                                         args.ptb_pct, args.batch_size, args.seed)
            depth = np.arange(nL) / (nL - 1)          # relative depth 0..1
            data[mid] = (depth, cka, cos)
            print(f"  {label}: {nL} layers, CKA {cka.min():.2f}-{cka.max():.2f}, cos {cos.min():.2f}-{cos.max():.2f}")
        meta = dict(k=args.k, n=args.n, ptb_type=args.ptb_type, ptb_pct=args.ptb_pct)
        np.savez(CACHE, data=data, meta=meta)
        print("cached ->", CACHE)

    # ---------- plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"Absolute vs RoPE: representation similarity under char@{meta['ptb_pct']}% "
                 f"(stripped k={meta['k']}, n={meta['n']})", fontsize=13)
    for mid, label, dtype, fam in MODELS:
        if mid not in data:
            continue
        depth, cka, cos = data[mid]
        ax1.plot(depth, cka, STYLE[fam], color=COLORS[mid], marker="o", ms=3, label=label)
        ax2.plot(depth, cka - cos, STYLE[fam], color=COLORS[mid], marker="o", ms=3, label=label)

    ax1.set(title="Stripped CKA (relational geometry)", xlabel="Relative depth (layer / final)",
            ylabel="stripped CKA", ylim=(0, 1.02))
    ax1.axhline(0.8, color="#2166ac", lw=0.5, ls=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right")
    ax2.set(title="CKA − cosine gap (unstrippable confound)", xlabel="Relative depth (layer / final)",
            ylabel="CKA − stripped cosine", ylim=(-0.1, 0.85))
    ax2.axhline(0, color="black", lw=0.4)
    ax2.legend(fontsize=8, loc="upper left")
    plt.savefig(f"{OUTDIR}/rope_vs_absolute.png", bbox_inches="tight", dpi=150)
    print("wrote", f"{OUTDIR}/rope_vs_absolute.png")

if __name__ == "__main__":
    main()
