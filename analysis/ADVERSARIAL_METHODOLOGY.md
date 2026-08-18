# Adversarial / Injection Perturbations — Methodology Notes

Companion notes for `adversarial_eval.py` and `src/adversarial_perturbs.py`.

## Context Insertion

Conditions: `clean`, `topic_shift`, `misleading_claim`, `adversarial`.

- `topic_shift` splices in a sentence sampled from a different, unrelated passage.
- `misleading_claim` corrupts the passage itself (mutates a year/number, or swaps
  two proper-noun-like tokens, or as a last resort negates a copula/aux verb) —
  no external claim bank or NER model is used.
- `adversarial` splices a placeholder span into the passage and optimizes it with
  a HotFlip-style gradient-guided attack (`src/hotflip.py`) to maximize the
  model's own NLL on the resulting sequence.

**PPL magnitude caveat**: perplexity here is computed per isolated, truncated
(≤128 token) passage, not via a sliding window over concatenated text. This
affects the absolute PPL magnitude (it will read higher than commonly-cited
GPT-2/WikiText-2 benchmarks) but not the relative comparison across conditions,
which is what this experiment is designed to measure.

## Question-Level Perturbations

Conditions: `clean`, `synonym`, `reorder`, `negation_paraphrase`, `adversarial_swap`.

Task: split each passage into `context` (all but the last word) and a held-out
`target_word`. Feed the (possibly perturbed) context to the model, greedily
generate a few tokens, and score a match via **loose substring search** for
`target_word` in the generated continuation.

**Task/metric mismatch caveat (important):** `adversarial_swap` optimizes the
whole-sequence NLL of the context — it has no notion of the held-out target
word at all, since the target word is never part of the input. Meanwhile
scoring only checks whether that one word shows up via substring match. These
two objectives are not aligned: a perturbation that makes the model's overall
predictions worse does not necessarily make it worse at guessing the one
specific held-out word, and can occasionally make the substring match easier
to satisfy by accident. **Treat any result where `adversarial_swap` scores
higher than `clean` as a harness artifact, not a robustness finding** — it
would indicate this mismatch, not that adversarial attacks improve accuracy.

**No confidence intervals:** none of these metrics have variance estimates
across seeds. Don't treat a single run as conclusive — rerun with several
`--seed` values before drawing conclusions.

**Dataset realism ≠ task realism:** the question-level task uses real
WikiText-2 passages, but "predict the passage's missing last word via
substring match" is a synthetic proxy task, not a validated QA benchmark. The
underlying text being real text does not make the task itself a realistic one.
Before trusting these results, consider swapping in a real QA benchmark
(e.g. SQuAD-style, exact-answer scoring) and an attack objective that
specifically targets the answer token rather than overall sequence loss.

## Metrics

| Metric | What it captures | Used in |
|---|---|---|
| Perplexity (`nll`, `perplexity` columns) / Δ PPL vs. clean | Overall language-modeling degradation | Context insertion |
| Next-token accuracy (`next_token_acc`) | Fraction of correctly predicted next tokens | Context insertion |
| Task accuracy (`match`, substring) | Whether the generated continuation contains the expected word | Question-level perturbation |

Not yet measured, but worth adding if this experiment gets extended further:
perturbation magnitude (tokens changed / perplexity of the inserted or swapped
span itself), attack efficiency (iterations/queries required to converge),
and cross-model transferability of the `adversarial` / `adversarial_swap`
conditions.
