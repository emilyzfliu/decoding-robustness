# Runs the HotFlip vs. uniform-random token-substitution comparison
# (main.py --ptb-type hotflip / --ptb-type token) across all 6 models in
# config.CROSS_MODEL_MODELS, at a fixed perturbation percentage.
#
# Run from the repo root, e.g. on a RunPod GPU instance:
#   bash scripts/hotflip_vs_token_sweep.sh
#
# Override the perturbation percentage or sample count via env vars:
#   PTB_PCT=30 N_SAMPLES=748 bash scripts/hotflip_vs_token_sweep.sh
#
# Resume-safe: main.py skips any --sample index already present in a
# model/ptb_type's evals.csv, so rerunning with a larger N_SAMPLES (or after
# an interrupted run) only computes what's missing.
#
# GPU to run on (override with: CUDA_VISIBLE_DEVICES=1 bash scripts/hotflip_vs_token_sweep.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PTB_PCT=${PTB_PCT:-30}
N_SAMPLES=${N_SAMPLES:-100}

cd "$(dirname "$0")/.."

MODELS=$(python3 -c "from config import CROSS_MODEL_MODELS; print(' '.join(CROSS_MODEL_MODELS))")

for model in $MODELS; do
    for ptb_type in token hotflip; do
        echo "=== $model / $ptb_type (pct=$PTB_PCT, n=$N_SAMPLES) ==="
        python main.py --model "$model" --ptb-type "$ptb_type" \
            --ptb-pct "$PTB_PCT" --n-samples "$N_SAMPLES"
    done
done
