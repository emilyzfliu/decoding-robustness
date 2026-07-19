"""
Cross-model comparison overlays (full sweeps, n=100). Reads each model's sweep
and overlays them plot-for-plot so GPT-2 / OPT / Llama / Qwen compare directly.

Produces, under figures/compare/:
  - behavioral_compare.png     3x3: [log ppl, output div, logit KL] x [char, token, shuffle]
  - representation_compare.png 2x3: [stripped CKA, stripped cosine] x [char, token, shuffle]
                               (vs RELATIVE depth, since models differ in #layers)

CPU-only; reads CSVs. Needs each model's sweep under --results-root/<dir>. GPT-2
must be re-run into /ssd_scratch first (its archive sweep is over quota).

Usage:
  python figures_compare.py
  python figures_compare.py --models opt_6.7b:OPT-6.7B,llama3_8b:Llama-3.1-8B
"""
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results-root", default="/ssd_scratch/varghese/decoding_robustness/results")
ap.add_argument("--models",
                default="gpt2:GPT-2,opt_6.7b:OPT-6.7B,llama3_8b:Llama-3.1-8B,qwen2.5_7b:Qwen2.5-7B",
                help="comma list of dir:label")
ap.add_argument("--out-dir", default="figures/compare")
args = ap.parse_args()

ROOT = args.results_root.rstrip("/")
MODELS = [tuple(x.split(":", 1)) for x in args.models.split(",")]   # (dir, label)
OUTDIR = args.out_dir
os.makedirs(OUTDIR, exist_ok=True)

# consistent styling with overlay.py: absolute solid, RoPE dashed
STYLE = {
    "gpt2":       dict(color="#999999", ls="-"),   # absolute
    "opt_6.7b":   dict(color="#d6604d", ls="-"),   # absolute
    "llama3_8b":  dict(color="#4393c3", ls="--"),  # RoPE
    "qwen2.5_7b": dict(color="#2166ac", ls="--"),  # RoPE
}
def sty(d):
    return STYLE.get(d, dict(color=None, ls="-"))

PTB = ["char", "token", "shuffle"]
NAMES = {"char": "Char substitution", "token": "Token substitution", "shuffle": "Token shuffling"}
REP_PCT = {"char": 25, "token": 25, "shuffle": 50}

def pcts(d, t):
    ds = glob.glob(f"{ROOT}/{d}/{t}/*")
    return sorted(int(os.path.basename(x)) for x in ds if os.path.basename(x).isdigit())

def load(d, t, p):
    return pd.read_csv(f"{ROOT}/{d}/{t}/{p}/evals.csv")

def nlayers(d):
    df = load(d, "char", REP_PCT["char"])
    return sum(c.startswith("activation_cka_layer_") for c in df.columns)

# ---------- Figure 1: behavioral (3 metrics x 3 ptb types) ----------
def logppl(df):
    v = df["perplexity"]
    return np.log(v[np.isfinite(v) & (v > 0)])

BMETRICS = [
    ("log perplexity", lambda df: np.median(logppl(df))),
    ("output divergence", lambda df: df["output_divergence"].median()),
    ("logit KL", lambda df: df["logit_kl"].median()),
]
fig, axs = plt.subplots(3, 3, figsize=(18, 13), sharex="col")
fig.suptitle("Cross-model behavioral comparison (median over 100 passages)", fontsize=15)
for c, t in enumerate(PTB):
    for r, (mname, fn) in enumerate(BMETRICS):
        ax = axs[r][c]
        for d, label in MODELS:
            ps = pcts(d, t)
            ys = [fn(load(d, t, p)) for p in ps]
            s = sty(d)
            ax.plot(ps, ys, s["ls"], color=s["color"], marker="o", ms=3, label=label)
        if r == 0:
            ax.set_title(NAMES[t], fontsize=12)
        if c == 0:
            ax.set_ylabel(mname, fontsize=11)
        if r == 2:
            ax.set_xlabel("Perturbation %")
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), fontsize=12,
           bbox_to_anchor=(0.5, -0.02))
plt.savefig(f"{OUTDIR}/behavioral_compare.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote", f"{OUTDIR}/behavioral_compare.png")

# ---------- Figure 2: representation (stripped CKA / cosine) vs relative depth ----------
RMETRICS = [
    ("Stripped CKA", "activation_cka_layer_"),
    ("Stripped cosine", "activation_cos_stripped_layer_"),
]
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
fig.suptitle("Cross-model representation similarity vs relative depth "
             "(char@25 / token@25 / shuffle@50)", fontsize=15)
for c, t in enumerate(PTB):
    for r, (mname, prefix) in enumerate(RMETRICS):
        ax = axs[r][c]
        for d, label in MODELS:
            df = load(d, t, REP_PCT[t])
            nL = nlayers(d)
            depth = np.arange(nL) / (nL - 1)
            ys = [df[f"{prefix}{L}"].mean() for L in range(nL)]
            s = sty(d)
            ax.plot(depth, ys, s["ls"], color=s["color"], marker="o", ms=2, label=label)
        ax.set_ylim(0, 1.02)
        if r == 0:
            ax.set_title(f"{NAMES[t]} @ {REP_PCT[t]}%", fontsize=12)
        if c == 0:
            ax.set_ylabel(mname, fontsize=11)
        if r == 1:
            ax.set_xlabel("Relative depth (layer / final)")
handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(MODELS), fontsize=12,
           bbox_to_anchor=(0.5, -0.02))
plt.savefig(f"{OUTDIR}/representation_compare.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote", f"{OUTDIR}/representation_compare.png")
