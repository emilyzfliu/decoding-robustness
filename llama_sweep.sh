#!/usr/bin/env bash
# Phase-1 Llama-3.1-8B sweep: behavioral + stripped-CKA/cosine representation
# metrics only (attention + ablation skipped). Mirrors the GPT-2 base sweeps.
#
# Usage:
#   HF_TOKEN=hf_... bash llama_sweep.sh
#   HF_TOKEN=hf_... CUDA_VISIBLE_DEVICES=2 BATCH=8 bash llama_sweep.sh
#
# Resumable: re-run to continue (completed samples are skipped).

source /ssd_scratch/miniconda3/etc/profile.d/conda.sh
conda activate decoding-robustness

export HF_HOME=/ssd_scratch/varghese/decoding_robustness/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Use the huggingface-cli login token (default location) unless HF_TOKEN is
# already set -- HF_HOME above otherwise relocates where HF looks for the token.
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

MODEL=meta-llama/Llama-3.1-8B
OUT=/ssd_scratch/varghese/decoding_robustness/results/llama3_8b
BATCH=${BATCH:-4}
LOG=/ssd_scratch/varghese/decoding_robustness/llama_sweep.log

cd /home/varghese/decoding-robustness

COMMON="--model $MODEL --dtype bfloat16 --no-attention --batch-size $BATCH --out-dir $OUT"

echo "=== llama sweep started $(date) on GPU ${CUDA_VISIBLE_DEVICES}, batch ${BATCH} ===" | tee "$LOG"
for i in $(seq 0 5 50);  do python main.py $COMMON --ptb-type char    --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done
for i in $(seq 5 5 50);  do python main.py $COMMON --ptb-type token   --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done
for i in $(seq 0 5 100); do python main.py $COMMON --ptb-type shuffle --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done
echo "=== llama sweep finished $(date) ===" | tee -a "$LOG"
