# How Different Input Perturbations Propagate Through GPT-2

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

Experiments are performed using GPT-2 small (`openai-community/gpt2`) to enable efficient CPU-based analysis and full-layer instrumentation.

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
src/evals.py     # All metrics
src/perturbs.py  # All perturbation implementations
main.py          # Base experiment eval loop
ablations.py     # Eval loop with single-head ablations
figures.py       # Generate all writeup figures
```

## Future Directions

Potential extensions include:

* more realistic perturbation distributions (OCR noise, typo distributions),
* stronger intervention-based analyses,
* cross-model comparison of perturbation-specific signatures.
* larger instruction-tuned models,
