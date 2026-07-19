#!/usr/bin/env bash
# Phase-1 cross-family sweep for the positional-encoding comparison:
#   OPT-6.7B     -- learned absolute positions (like GPT-2)
#   Qwen2.5-7B   -- RoPE (like Llama)
# Both ungated (no HF_TOKEN needed). Behavioral + stripped-cosine/CKA metrics
# only (attention/ablation skipped). Mirrors llama_sweep.sh.
#
# Usage:
#   bash sweep.sh                          # both models
#   bash sweep.sh --only qwen2.5_7b        # just one (by its short name)
#   bash sweep.sh --only opt_6.7b
#   CUDA_VISIBLE_DEVICES=2 BATCH=8 bash sweep.sh --only qwen2.5_7b
#
# Resumable: re-run to continue (completed samples are skipped).

ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

source /ssd_scratch/miniconda3/etc/profile.d/conda.sh
conda activate decoding-robustness

export HF_HOME=/ssd_scratch/varghese/decoding_robustness/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH=${BATCH:-4}
ROOT=/ssd_scratch/varghese/decoding_robustness/results

cd /home/varghese/decoding-robustness

run_model () {
  local model="$1" name="$2"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
    echo "skipping $name (--only $ONLY)"; return
  fi
  local out="$ROOT/$name"
  local log="/ssd_scratch/varghese/decoding_robustness/${name}_sweep.log"
  local common="--model $model --dtype bfloat16 --no-attention --batch-size $BATCH --out-dir $out"
  echo "=== $name ($model) started $(date) on GPU ${CUDA_VISIBLE_DEVICES}, batch ${BATCH} ===" | tee "$log"
  for i in $(seq 0 5 50);  do python main.py $common --ptb-type char    --ptb-pct "$i" 2>&1 | tee -a "$log"; done
  for i in $(seq 5 5 50);  do python main.py $common --ptb-type token   --ptb-pct "$i" 2>&1 | tee -a "$log"; done
  for i in $(seq 0 5 100); do python main.py $common --ptb-type shuffle --ptb-pct "$i" 2>&1 | tee -a "$log"; done
  echo "=== $name finished $(date) ===" | tee -a "$log"
}

run_model facebook/opt-6.7b  opt_6.7b
run_model Qwen/Qwen2.5-7B     qwen2.5_7b
