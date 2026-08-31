# Runs the position-coupled HotFlip vs. random-control comparison
# (run_paired_hotflip.py) across all 6 models in config.CROSS_MODEL_MODELS,
# at a given perturbation percentage. Unlike the plain hotflip/token dirs
# from hotflip_vs_token_sweep.sh, this pipeline guarantees the random
# control touches exactly the same positions HotFlip did -- see
# src/perturbs.py:hotflip_and_coupled_random and run_paired_hotflip.py.
#
# Run from the repo root, e.g. on a RunPod GPU instance:
#   bash scripts/paired_hotflip_sweep.sh
#
# Override the percentage or sample count via env vars:
#   PTB_PCT=30 N_SAMPLES=300 bash scripts/paired_hotflip_sweep.sh
#
# NOTE on resume: same rationale as hotflip_vs_token_sweep.sh -- a single
# run_paired_hotflip.py invocation always regenerates the full attack for
# every text passed to it, it doesn't skip recomputation on its own. So
# this script checks BOTH hotflip_paired/ and token_coupled/ evals.csv for
# completeness before invoking the script, and skips entirely when a model
# is already done. A model that's only partially done (crash mid-write)
# still gets a full re-run, since the underlying script can't resume
# generation partway through either.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PTB_PCT=${PTB_PCT:-30}
N_SAMPLES=${N_SAMPLES:-300}

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

for model in $MODELS; do
    hf_csv="results/$model/hotflip_paired/$PTB_PCT/evals.csv"
    tc_csv="results/$model/token_coupled/$PTB_PCT/evals.csv"
    if is_complete "$hf_csv" && is_complete "$tc_csv"; then
        echo "=== $model (pct=$PTB_PCT, n=$N_SAMPLES) === already complete, skipping"
        continue
    fi
    echo "=== $model (pct=$PTB_PCT, n=$N_SAMPLES) ==="
    python run_paired_hotflip.py --model "$model" --ptb-pct "$PTB_PCT" --n-samples "$N_SAMPLES"
done
