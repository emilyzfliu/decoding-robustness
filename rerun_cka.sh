#!/usr/bin/env bash
# Re-run the base perturbation sweeps to populate the new stripped-CKA /
# stripped-cosine representation metrics. Ablations are NOT re-run (they are
# output_only and unaffected). Originals are backed up in results_backup_pre_cka/.
#
# Usage:
#   bash rerun_cka.sh                 # runs on GPU 0
#   CUDA_VISIBLE_DEVICES=2 bash rerun_cka.sh
#   nohup bash rerun_cka.sh &         # detached
#
# Resumable: if interrupted, just run it again (completed samples are skipped).

source /ssd_scratch/miniconda3/etc/profile.d/conda.sh
conda activate decoding-robustness

export HF_HOME=/archive/varghese/decoding-robustness/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd /home/varghese/decoding-robustness

LOG=/archive/varghese/decoding-robustness/rerun_cka.log
echo "=== re-run started $(date) on GPU ${CUDA_VISIBLE_DEVICES} ===" | tee "$LOG"

for i in $(seq 0 5 50);  do python main.py --ptb-type char    --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done
for i in $(seq 5 5 50);  do python main.py --ptb-type token   --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done
for i in $(seq 0 5 100); do python main.py --ptb-type shuffle --ptb-pct "$i" 2>&1 | tee -a "$LOG"; done

echo "=== re-run finished $(date) ===" | tee -a "$LOG"
