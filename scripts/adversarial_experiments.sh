# Runs the adversarial/injection perturbation experiments
# (analysis/adversarial_eval.py) across all 6 local-scale models.
#
# Run from the repo root, e.g. on a RunPod GPU instance:
#   bash scripts/adversarial_experiments.sh
#
# Optionally set HF_HOME to control where model/dataset downloads are cached:
#   export HF_HOME=/path/to/hf_cache
#
# GPU to run on (override with: CUDA_VISIBLE_DEVICES=1 bash scripts/adversarial_experiments.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$(dirname "$0")/.."

for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl' 'qwen2.5_0.5b' 'qwen2.5_1.5b'; do
    python analysis/adversarial_eval.py --model $model --experiment both
done
