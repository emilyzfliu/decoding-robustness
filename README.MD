# Input Perturbations Propagate Differently: A Mechanistic Analysis of GPT-2 Robustness

Mechanistic failure modes under lexical, tokenizer, and positional corruption.

Most robustness work treats language models as black boxes and evaluates only output quality under noisy inputs. This project instead takes a mechanistic approach, tracing how different perturbation types propagate internally through GPT-2.

Rather than treating corruption as a single scalar “noise” variable, this project separates three distinct corruption interfaces:

* **Character substitution** → tokenizer disruption
* **Token substitution** → lexical corruption
* **Token shuffling** → positional corruption

The project analyzes how these perturbations affect:

* output and logit behavior,
* hidden-state representations,
* attention entropy patterns,
* and the causal importance of specific attention heads.

## Model

The primary local experiments use GPT-2 small (`openai-community/gpt2`) to enable efficient CPU-based analysis and full-layer instrumentation. A separate GPT-2 Large robustness sweep runs through the Kaggle kernels under `kaggle/gpt2-large/`.

## Dataset

Experiments use the cleaned `wikitext-2-raw-v1` test split.

Input sequences:

* shorter than 128 tokens are discarded,
* remaining sequences are truncated to 128 tokens.

## Perturbation Types

### Character Substitution

Random character-level replacement applied directly to raw text prior to tokenization.

Primarily probes:

* tokenizer disruption,
* subword fragmentation,
* instability in lexical representations.

### Token Substitution

Random token replacement sampled uniformly from the tokenizer vocabulary.

Primarily probes:

* lexical corruption,
* semantic degradation,
* corrupted token identity.

### Word Substitution

Random whole-word replacement sampled from a word-level vocabulary derived from the tokenizer.

Primarily probes:

* lexical corruption at the word granularity,
* tokenization-independent semantic degradation.

### Typo Substitution

Realistic character-level typos sampled from QWERTY keyboard adjacency (plus common punctuation typos).

Primarily probes:

* tokenizer disruption under plausible (non-uniform) noise,
* how realistic typos differ from uniform character substitution.

### Synonym Substitution

Word replacement using WordNet synonyms (same synset members), preserving part-of-speech and case.

Primarily probes:

* semantic-preserving lexical corruption,
* whether meaning-preserving substitutions are "easier" than random word replacement.

### Token Shuffling

Random permutation within a contiguous local token window.

Primarily probes:

* positional corruption,
* local syntactic disruption,
* attention robustness to reordered context.

## Metrics

### Behavioral Metrics

* Sequence-level negative log likelihood (NLL)
* Output divergence
* Logit KL divergence

### Representation Metrics

* Layer-wise activation cosine similarity
* Linear CKA (stripped: top-variance dims removed)
* TwoNN intrinsic dimension (manifold expansion/collapse)
* MKNN intrinsic dimension (maximum-likelihood estimate)
* Attention entropy dynamics

### Causal Analysis

* Targeted attention-head ablations
* Gap reduction analysis under perturbation

## Core Findings

* Different perturbation types produce qualitatively different internal failure modes.
* Character-level perturbations produce abrupt early representational collapse.
* Token substitution produces smoother semantic degradation.
* Local token shuffling produces comparatively gradual degradation despite positional corruption.
* Attention entropy shifts alone are insufficient to determine causal importance.
* A small number of late-layer heads disproportionately propagate corrupted representations under character perturbation.

## Repository

```text
src/evals.py                        # All metrics (incl. CKA, TwoNN, MKNN)
src/perturbs.py                     # All perturbation implementations
main.py                             # Base experiment eval loop
ablations.py                        # Eval loop with single-head ablations
figures.py                          # Generate all writeup figures
compare_bpe_vs_word.py              # BPE vs word-level substitution experiment
plot_comparison.py                  # Figures for the BPE vs word comparison
compare_perturbation_types.py       # Cross-type CKA comparison across 6 ptbs
validate_intrinsic_dim.py           # Convergence/bootstrap validation of TwoNN/MKNN
test_quick.py                       # Smoke-test suite for perturbations & metrics
```

Results live under `results/{model}/{ptb_type}/{pct}/evals.csv` (the layout written
by `main.py`); `--model`, `--n-samples`, and `--batch-size` control which model,
how many sequences, and the per-step batch size. The BPE-vs-word comparison uses
its own self-contained layout under `results/compare_bpe_word/`.

## Future Directions

Potential extensions include:

* more realistic perturbation distributions (OCR noise, typo distributions),
* stronger intervention-based analyses,
* cross-model comparison of perturbation-specific signatures.
* larger instruction-tuned models,
