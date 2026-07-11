# Perturbation Robustness in GPT-2 — Findings

*GPT-2 small (124M) · WikiText-2 test, 100 passages × 128 tokens · clean-vs-perturbed comparison.*

Two outcomes: (A) how three corruption types propagate internally, and (B) a methodological finding — the representation-similarity metric was confounded by GPT-2's massive activations, and had to be fixed.

---

## Setup

Each clean passage is run through GPT-2 twice — clean and perturbed — and compared. Three perturbation types, each isolating one corruption interface:

| Type | Operation | Interface probed |
|---|---|---|
| **char** | random ASCII character swaps in raw text | tokenizer disruption |
| **token** | random token substitutions from vocab | lexical / semantic corruption |
| **shuffle** | reorder words within a local window | positional corruption |

**Metrics:** behavioral (perplexity, output divergence, logit-KL), representational (per-layer similarity), causal (single-head ablations).

---

## A. Behavioral findings — three distinct signatures

| | Onset (div @ low noise) | Perplexity behavior | Character |
|---|---|---|---|
| **char** | **0.55 @ 5%** → 0.93 @ 50% | peaks ~1660 @ 20%, then *declines* | front-loaded, saturating |
| **token** | 0.17 @ 5% → 0.68 @ 50% | superlinear: 170 → **30,516** | gradual output, explosive "surprise" |
| **shuffle** | 0.10 @ 5% → 0.70 @ 100% | steady: 57 → 1537 | gentle, near-linear |

- **Character noise is front-loaded** — 5% corruption already flips half the output tokens (it re-fragments tokenization at the input).
- **Token substitution** barely changes greedy output at first but destroys likelihood fastest — valid-but-wrong tokens are maximally surprising.
- **Shuffling is the gentlest** — the model is strikingly robust to local word-order scrambling.
- **Curiosity:** char perplexity is *non-monotonic* (peaks at 20%, then falls) — past a threshold, text becomes uniformly random and the model stops being confidently wrong. Divergence and logit-KL keep rising, so it's perplexity-specific.

*(% units are not directly comparable across types: per-character vs per-token vs window-fraction. Compare shapes/mechanisms, not crossing points.)*

---

## B. Representational findings — per-layer similarity (corrected metric)

**Stripped CKA** (top-5 outlier dims removed; see §D), clean-vs-perturbed, at mid-corruption (char@25 / token@25 / shuffle@50):

| Layer | char | token | shuffle |
|---|---|---|---|
| L0 (embedding) | 0.86 | 0.90 | 0.88 |
| L3–8 (middle) | ~0.44 | ~0.70 | ~0.62 |
| L11 | 0.39 | 0.66 | 0.60 |
| **L12 (final)** | **0.10** | **0.35** | **0.40** |

- **Similarity declines into the final layer** — no recovery. (Raw cosine falsely showed ~0.95 recovery at L12; that was an artifact — see §D.)
- **Severity ordering holds at every layer**: char (most corrupted) < shuffle < token, matching behavioral rankings.
- **char is hit hardest throughout**, consistent with its input-level (tokenizer) disruption. The embedding-layer gap is clearest in per-token cosine (char 0.64 vs token/shuffle ~0.82).
- Stripped cosine and stripped CKA **agree in shape and ordering** — the validation that the confound is gone.

---

## C. Causal findings — head ablations (a null result)

Zeroing the top entropy-slope-sensitive heads barely changes the corruption gap:

| Perturbation | Best single-head gap reduction (logit-KL) |
|---|---|
| char @ 50% | **−2.6%** (L11 H7) |
| token @ 50% | −0.5% |
| shuffle @ 100% | −0.1% |

- **Every ablation < 2.7%; most < 1%.** Attention-entropy sensitivity is **not** a reliable indicator of causal importance.
- Corruption propagation appears **diffuse / redundant**, not concentrated in a few heads. (The original "small number of late-layer heads dominate" hypothesis is not supported; at most a weak late-layer signal for char.)

---

## D. Methodological finding — a confounded similarity metric

**Symptom:** raw cosine similarity "recovered" to ~0.95 at the final layer even when outputs were completely different — a red flag.

**Cause:** GPT-2's residual stream develops a few **massive-activation "rogue" dimensions** (max magnitude ~200 vs median 0.33; one dim ≈ 90% of a layer's variance). They implement **attention sinks** (the first token absorbs ~44% of all attention) and act as a learned bias/anchor. They are near-constant and survive perturbation, so any similarity metric they dominate reports false "similarity."

**Every off-the-shelf metric is fooled — differently:**

| Metric | Weights dims by | Fooled by | Result |
|---|---|---|---|
| raw cosine | magnitude (incl. offset) | the huge **offset** | falsely **high** (0.95) |
| z-scored cosine | 1/std | flat dims' **amplified noise** | falsely **low** (0.04) |
| plain CKA | variance | huge preserved **positional variance** | falsely **high** (0.999) |

**Fix:** drop the top-k (k=5) highest-variance dims per layer, *then* measure. Once the outliers are gone, cosine and CKA converge (~0.43 mid-layer) — the honest content signal. Implemented in `eval.py::activation_cka` (reports stripped CKA + stripped cosine per sample); raw cosine columns retained for reference.

**Architecture note (why this happens):** GPT-2 is pre-LayerNorm — LayerNorm normalizes each sub-layer's *input* (a branched copy), not the residual stream itself, and not the sub-layer *output*. The stream is an un-normalized running sum, so a few dimensions accumulate huge values (stream norm grows ~5 → ~3000 across layers) that LayerNorm strips only at read-time. `k=5` is a hyperparameter; qualitative conclusions are stable, but a k∈{3,5,10} sweep is worth doing before publishing.

---

## Reproduction

- **Env:** conda env `decoding-robustness` (Python 3.11, torch cu128). GPU: `CUDA_VISIBLE_DEVICES` (defaults to 0).
- **Data/cache & results:** `/archive/varghese/decoding-robustness/` (`HF_HOME` + `results/` symlink).
- **Run:** `bash experiments.sh` (base sweeps + ablations) or `bash rerun_cka.sh` (base sweeps with corrected metrics). `python figures.py` for plots.
- Backup of pre-CKA results: `results_backup_pre_cka/`.

## Open follow-ups

- k-sensitivity sweep for the stripped metric.
- Representation-level ablation (run ablations with CKA on) to test causal restoration of representational similarity.
- Investigate the char-perplexity non-monotonicity.
