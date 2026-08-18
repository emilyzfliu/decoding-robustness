# Adversarial / Injection Perturbations — Results

GPT-2 Small (117M), WikiText-2 (`wikitext-2-raw-v1`, test split, 748 passages passing the
>128-word filter), seed=1. Two sequence-length configurations are reported: the original
`max_length=128` (full 748-passage run) and a `max_length=1024` config that lets nearly
every passage run its full natural length with no truncation (currently a 50-sample
pilot; full 748-passage run pending). See `ADVERSARIAL_METHODOLOGY.md` for full caveats.

## 1. Context Insertion

Injecting irrelevant, misleading, or adversarially-optimized text into a passage and
measuring the impact on perplexity (PPL) and next-token prediction accuracy.

| Condition | PPL | Δ PPL vs. clean | Next-token acc. |
|---|---|---|---|
| Clean | 50.37 | — | 31.8% |
| Misleading factual claim | 52.57 | +2.20 | 31.5% |
| Topic-shift (irrelevant sentence) | 66.38 | +16.01 | 29.9% |
| Adversarial (gradient-guided) | 243.84 | +193.47 | 26.8% |

**Key finding:** every perturbation type increases PPL and reduces accuracy relative to
clean context, and the effect scales with how deliberately the perturbation is crafted.
The adversarially-optimized insertion increases perplexity ~5x — a far larger effect than
either naturally-occurring distractor condition, showing the model is disproportionately
vulnerable to targeted corruption versus generic noise.

### Example transformations

- **Topic-shift**: splices a sentence from an unrelated passage into the middle of the text.
  > "The Eiffel Tower was completed in 1889 and remains one of **[A separate unrelated
  > story about a chess tournament held in Reykjavik last spring drew large crowds.]** the
  > most visited monuments in Paris, attracting millions of tourists every year."

- **Misleading factual claim**: corrupts a number/date already in the passage (falls back
  to swapping two proper nouns, then negating a copula, if no number is present).
  > "The Eiffel Tower was completed in **1889**..." → "...completed in **1778**..."

- **Adversarial (gradient-guided)**: splices a placeholder span into the passage and
  optimizes its tokens via a HotFlip-style attack to maximize the model's own NLL.
  Optimizes purely for loss, not fluency, hence the incoherent output:
  > "The quick brown fox jumps over the lazy **the Honolulublance emb** dog in the year
  > 1995 near Paris..."

### 1a. Context Insertion — max_length=1024 (untruncated context, n=50 pilot)

Same conditions, but with the sequence-length cap raised from 128 to GPT-2's real
1024-token limit. Since the >128-word filter guarantees ≥128 tokens of content but not
1024, most passages now run their full natural length (mean ~226 tokens, max 542 —
essentially none get truncated) rather than being cut off at 128. **This is a 50-sample
pilot, not yet the full 748-passage run** — treat magnitudes as preliminary.

| Condition | PPL | Δ PPL vs. clean | Next-token acc. |
|---|---|---|---|
| Clean | 43.26 | — | 34.0% |
| Misleading factual claim | 44.88 | +1.62 | 33.8% |
| Topic-shift (irrelevant sentence) | 51.78 | +8.52 | 32.9% |
| Adversarial (gradient-guided) | 99.96 | +56.70 | 31.3% |

Same qualitative ordering as the 128-token result (clean < misleading_claim < topic_shift
≪ adversarial), and next-token accuracy is slightly *higher* across the board than the
128-token version — consistent with the model benefiting from fuller, untruncated
context. Note the Δ PPL magnitudes are smaller here than at 128 tokens (e.g. adversarial
+56.7 vs. +193.5) — worth confirming this pattern holds once the full 748-passage run
completes, since it's currently only 50 samples.

## 2. Question-Level Perturbations

Rephrasing or adversarially modifying a passage's context (holding out its final word),
measured via a proxy next-word-prediction task: does the model's greedy continuation
contain the held-out word (loose substring match)?

| Condition | Accuracy |
|---|---|
| Clean | 2.01% |
| Negation-preserving paraphrase | 2.14% |
| Syntactic reordering | 2.14% |
| Synonym substitution | 2.27% |
| Adversarial (gradient-guided word swap) | 1.60% |

**Key finding:** unlike context insertion, none of these perturbations produce a
meaningful accuracy change — all five conditions cluster within a ~0.7-point band. This
task is intrinsically hard (guessing the *exact* final word of an arbitrary ~128-word
passage via substring match), so absolute accuracy is low across the board; see the
methodology notes before reading anything into the ordering at this sample size.

### Example transformations

- **Synonym substitution**: WordNet-based, POS-unaware, case-preserving single-word swaps.
  > "...jumps over the lazy **dog** in the year 1995 **near**..." → "...jumps over the lazy
  > **tag** in the year 1995 **approxima**[tely]..."

- **Syntactic reordering**: swaps two adjacent comma/conjunction-delimited clauses.
  > "The committee reviewed the proposal, **and** although several members raised
  > concerns, the board did not reject the plan..." → "**although several members raised
  > concerns, and** The committee reviewed the proposal, the board did not reject the
  > plan..."

- **Negation-preserving paraphrase**: contraction ↔ expanded-form rewrite, polarity intact.
  > "...the board **did not** reject the plan..." → "...the board **didn't** reject the
  > plan..."

- **Adversarial (gradient-guided word swap)**: HotFlip attack over the context tokens only
  (the held-out word is never part of the input, so it can't be directly attacked).
  > "...jumps over the lazy dog in the year 1995 near..." → "**Nik murm faded**
  > **externalActionCode�BACKiHUD Phantom Barcl LLrils**..."

## 3. Metrics used

| Metric | What it captures | Used in |
|---|---|---|
| Perplexity (PPL) / Δ PPL | Overall language-modeling degradation under perturbation | Context insertion |
| Next-token accuracy | Fraction of correctly predicted next tokens | Context insertion |
| Task accuracy (substring match) | Whether the model's continuation contains the expected word | Question-level perturbation |

## 4. Caveats (see ADVERSARIAL_METHODOLOGY.md for full detail)

- **Task/metric mismatch**: `adversarial_swap` optimizes whole-sequence loss, not the
  specific held-out word — treat any surprising ordering in the question-level table as a
  harness artifact, not a robustness finding.
- **No confidence intervals**: single run, seed=1 — rerun with multiple seeds before
  treating the question-level ordering as conclusive.
- **Absolute PPL magnitude**: computed per isolated, truncated passage rather than via a
  sliding window over concatenated text, so it reads higher than commonly-cited GPT-2/
  WikiText-2 benchmarks (~24-29). This doesn't affect the relative comparison across
  conditions.
- **Dataset realism ≠ task realism**: question-level task uses real WikiText-2 text, but
  "predict the passage's missing last word via substring match" is a synthetic proxy, not
  a validated QA benchmark.
