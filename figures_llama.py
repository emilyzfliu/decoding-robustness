"""
Standalone plotting for the Llama-3.1-8B Phase-1 sweep:
  - behavioral panels (log perplexity, output divergence, logit KL)
  - representation panels: stripped cosine ("content") vs stripped CKA ("geometry")

Attention/ablation panels are omitted (Phase 1 skips them). CPU-only; reads the
sweep CSVs and writes PNGs to figures/llama3_8b/.

Usage: python figures_llama.py
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--with-input-div", action="store_true",
                help="also plot input-vs-output divergence (needs the tokenizer + HF auth)")
ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
args = ap.parse_args()

RESULTS = "/ssd_scratch/varghese/decoding_robustness/results/llama3_8b"
OUTDIR = "figures/llama3_8b"
os.makedirs(OUTDIR, exist_ok=True)

PTB = ["char", "token", "shuffle"]
NAMES = {"char": "Char substitution", "token": "Token substitution", "shuffle": "Token shuffling"}
REP_PCT = {"char": 25, "token": 25, "shuffle": 50}   # representative mid-corruption level

def pcts(t):
    ds = glob.glob(f"{RESULTS}/{t}/*")
    return sorted(int(os.path.basename(d)) for d in ds if os.path.basename(d).isdigit())

def load(t, p):
    return pd.read_csv(f"{RESULTS}/{t}/{p}/evals.csv")

def band(ax, xs, series, label):
    med = [np.median(s) for s in series]
    lo = [np.percentile(s, 25) for s in series]
    hi = [np.percentile(s, 75) for s in series]
    ax.plot(xs, med, marker="o", ms=3, label=label)
    ax.fill_between(xs, lo, hi, alpha=0.2)

# ---------- Figure 1: behavioral ----------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Llama-3.1-8B — Behavioral metrics", fontsize=14)
for t in PTB:
    ps = pcts(t)
    dfs = [load(t, p) for p in ps]
    logppl = [np.log(d["perplexity"][np.isfinite(d["perplexity"]) & (d["perplexity"] > 0)]) for d in dfs]
    band(axs[0], ps, logppl, NAMES[t])
    band(axs[1], ps, [d["output_divergence"] for d in dfs], NAMES[t])
    band(axs[2], ps, [d["logit_kl"] for d in dfs], NAMES[t])
axs[0].set(title="log perplexity (seq-level NLL)", xlabel="Perturbation %", ylabel="log ppl"); axs[0].legend()
axs[1].set(title="Output divergence", xlabel="Perturbation %"); axs[1].legend()
axs[2].set(title="Logit KL divergence", xlabel="Perturbation %"); axs[2].legend()
plt.savefig(f"{OUTDIR}/behavioral.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ---------- Figure 2: representation (content vs geometry) ----------
sample = load("char", REP_PCT["char"])
nL = sum(c.startswith("activation_cka_layer_") for c in sample.columns)
layers = list(range(nL))

fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.suptitle("Llama-3.1-8B — Representation similarity to clean, per layer", fontsize=14)
for i, t in enumerate(PTB):
    df = load(t, REP_PCT[t])
    cos = [df[f"activation_cos_stripped_layer_{L}"].mean() for L in layers]
    cka = [df[f"activation_cka_layer_{L}"].mean() for L in layers]
    axs[i].plot(layers, cos, marker="o", ms=3, color="#d6604d", label="content (stripped cosine)")
    axs[i].plot(layers, cka, marker="^", ms=3, color="#2166ac", label="geometry (stripped CKA)")
    axs[i].set(title=f"{NAMES[t]} @ {REP_PCT[t]}%", xlabel="Layer", ylim=(0, 1.02))
    axs[i].legend(fontsize=9)
axs[0].set_ylabel("Similarity to clean")
plt.savefig(f"{OUTDIR}/representation.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print("wrote", f"{OUTDIR}/behavioral.png", "and", f"{OUTDIR}/representation.png")

# ---------- Figure 3: metric heatmaps (layer x perturbation %) ----------
METRICS = [
    ("Raw cosine", "activation_cos_sim_layer_"),
    ("Stripped cosine", "activation_cos_stripped_layer_"),
    ("Stripped CKA", "activation_cka_layer_"),
]
fig, axs = plt.subplots(3, 3, figsize=(20, 13))
fig.suptitle("Llama-3.1-8B — Representation similarity to clean (layer x perturbation %)", fontsize=15)
im = None
for c, t in enumerate(PTB):
    ps = pcts(t)
    dfs = [load(t, p) for p in ps]
    for r, (mname, prefix) in enumerate(METRICS):
        M = np.array([[d[f"{prefix}{L}"].mean() for L in range(nL)] for d in dfs])
        ax = axs[r][c]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(ps))); ax.set_yticklabels(ps, fontsize=6)
        xt = list(range(0, nL, 4)); ax.set_xticks(xt); ax.set_xticklabels([f"L{L}" for L in xt], fontsize=7)
        if r == 0:
            ax.set_title(NAMES[t], fontsize=12)
        if c == 0:
            ax.set_ylabel(f"{mname}\nPerturbation %", fontsize=10)
        if r == 2:
            ax.set_xlabel("Layer")
fig.colorbar(im, ax=axs, shrink=0.6, label="similarity to clean")
plt.savefig(f"{OUTDIR}/metric_heatmaps.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote", f"{OUTDIR}/metric_heatmaps.png")

# ---------- Figure 4 (optional): input vs output divergence ----------
# Needs the tokenizer (the `token` perturbation decodes random ids), so this
# panel requires HF auth -- hence the --with-input-div flag.
if args.with_input_div:
    import random
    import Levenshtein
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from src.perturbs import perturb

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    texts = [x for x in ds["test"]["text"] if len(x.split()) > 128]
    texts = random.Random(1).sample(texts, 100)           # same sampling as the sweep

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Llama-3.1-8B — Input vs Output divergence", fontsize=14)
    for i, t in enumerate(PTB):
        ps = pcts(t)
        indiv = []
        for p in ps:
            pert = perturb(texts, p, random.Random(1), t, tok)   # same seed as main.py
            indiv.append([Levenshtein.distance(a, b) / max(len(a), len(b)) for a, b in zip(texts, pert)])
        outdiv = [load(t, p)["output_divergence"] for p in ps]
        band(axs[i], ps, indiv, "input divergence")
        band(axs[i], ps, outdiv, "output divergence")
        axs[i].set(title=NAMES[t], xlabel="Perturbation %")
        axs[i].legend()
    axs[0].set_ylabel("Divergence")
    plt.savefig(f"{OUTDIR}/in_vs_out.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("wrote", f"{OUTDIR}/in_vs_out.png")
