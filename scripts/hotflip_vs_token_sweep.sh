# Runs the HotFlip vs. uniform-random token-substitution comparison
# (main.py --ptb-type hotflip / --ptb-type token) across all 6 models in
# config.CROSS_MODEL_MODELS, at low/medium/high perturbation percentages —
# matching the paper's Figure 2/3 convention of 5% / 30% / 50%.
#
# Run from the repo root, e.g. on a RunPod GPU instance:
#   bash scripts/hotflip_vs_token_sweep.sh
#
# Override the percentage list or sample count via env vars (space-separated
# for multiple percentages):
#   PTB_PCTS="5 30 50" N_SAMPLES=748 bash scripts/hotflip_vs_token_sweep.sh
#
# NOTE on resume: main.py's own --sample dedup only prevents duplicate CSV
# rows — it does NOT skip recomputation. Every batch still runs a full
# forward pass, and hotflip's perturb() call always regenerates the entire
# attack for every text in the run, even if that file is already complete.
# So this script checks completeness itself (evals.csv already has
# N_SAMPLES unique samples) BEFORE invoking main.py, and skips the call
# entirely when a combo is already done — that's what actually avoids
# redoing finished work. A combo that's partially done (like a file left
# mid-write after a crash) still gets a full main.py invocation, since
# main.py itself can't resume generation partway through.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PTB_PCTS=${PTB_PCTS:-"5 30 50"}
N_SAMPLES=${N_SAMPLES:-100}

cd "$(dirname "$0")/.."

MODELS=$(python3 -c "from config import CROSS_MODEL_MODELS; print(' '.join(CROSS_MODEL_MODELS))")

is_complete() {
    # $1 = path to evals.csv
    python3 -c "
import pandas as pd, sys
try:
    df = pd.read_csv('$1', usecols=['sample'])
    sys.exit(0 if df['sample'].nunique() >= $N_SAMPLES else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

for pct in $PTB_PCTS; do
    for model in $MODELS; do
        for ptb_type in token hotflip; do
            csv_path="results/$model/$ptb_type/$pct/evals.csv"
            if is_complete "$csv_path"; then
                echo "=== $model / $ptb_type (pct=$pct, n=$N_SAMPLES) === already complete, skipping"
                continue
            fi
            echo "=== $model / $ptb_type (pct=$pct, n=$N_SAMPLES) ==="
            python main.py --model "$model" --ptb-type "$ptb_type" \
                --ptb-pct "$pct" --n-samples "$N_SAMPLES"
        done
    done
done
