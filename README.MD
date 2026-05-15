# LLM Decoding Robustness

How do transformer internals respond to simple input perturbations?
Most robustness work treats models as black boxes and measures output quality.
This project takes a mechanistic approach, using attention analysis, hidden state tracing, and targeted ablations to identify the degradations of specific model components.

## Model
This project uses the gpt2 model on huggingface (`openai-community/gpt2`) to be small enough to run on CPU. Support for larger models may be included in the future.

## Dataset
Cleaned `wikitext-2-raw-v1` test set

## Perturbations
Evaluates the following perturbations (for now)
- Random character substitution
- Random token substitution
- Token shuffling

## Metrics
### Output
- Perplexity
- Output Divergence
- Stability