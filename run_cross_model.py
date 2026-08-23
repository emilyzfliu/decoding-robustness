"""
Serial driver for the cross-model robustness experiment.

Runs `main.py` for every model x perturbation type over the full percentage
grid (5..50 step 5, plus the char/0 baseline), 300 samples each, writing to
`--out-root` (default: results_v2 so existing results/ is untouched).

Resume-safe: skips (model, ptb_type) combinations whose percentage dirs
already contain complete evals.csv files (100+ samples, full grid covered).

Usage:
    python run_cross_model.py                          # all 6 local models
    python run_cross_model.py --models gpt2,qwen2.5_0.5b
    python run_cross_model.py --models qwen2.5_7b --out-root results_v2   # e.g. on a server
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

from config import CROSS_MODEL_MODELS

PTB_TYPES = ['char', 'token', 'word', 'shuffle', 'typo', 'synonym', 'adv']
PCTS = [x * 5 for x in range(1, 11)]
GRID_PCTS = {'char': [0] + PCTS, 'shuffle': PCTS, 'token': PCTS,
             'word': PCTS, 'typo': PCTS, 'synonym': PCTS, 'adv': PCTS}
N_SAMPLES = 300
# TARGET_ROWS = N_SAMPLES * 127  # 300 sequences x 127 token positions


def model_dir_complete(out_root, model, ptb_type, n_samples=N_SAMPLES):
    """True if every percentage dir has a complete evals.csv for this type."""
    for pct in GRID_PCTS[ptb_type]:
        path = os.path.join(out_root, model, ptb_type, str(pct), 'evals.csv')
        if not os.path.exists(path):
            return False
        with open(path) as f:
            n_rows = sum(1 for _ in f) - 1
        if n_rows < n_samples:
            return False
    return True


def log(msg, log_path):
    line = f'[{datetime.datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_one(model, ptb_type, out_root, n_samples, log_path):
    log(f'--- {model} / {ptb_type} (n={n_samples}, pcts={GRID_PCTS[ptb_type]}) ---', log_path)
    t0 = time.time()
    cmd = [sys.executable, 'main.py',
           '--model', model,
           '--ptb-type', ptb_type,
           '--n-samples', str(n_samples),
           '--out-root', out_root]
    with open(log_path, 'a', encoding='utf-8') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    if proc.returncode != 0:
        log(f'!!! FAILED {model}/{ptb_type} (exit {proc.returncode})', log_path)
    else:
        log(f'    done {model}/{ptb_type} in {dt:.1f} min', log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=','.join(CROSS_MODEL_MODELS),
                        help='Comma-separated model keys (default: all 6 local models)')
    parser.add_argument('--ptb-types', default=','.join(PTB_TYPES))
    parser.add_argument('--out-root', default='results_v2')
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES)
    parser.add_argument('--force', action='store_true', help='Rerun even if dirs look complete')
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, 'run_cross_model.log')
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    ptb_types = [t.strip() for t in args.ptb_types.split(',') if t.strip()]

    total_est = 0
    for model in models:
        for ptb_type in ptb_types:
            if not args.force and model_dir_complete(args.out_root, model, ptb_type):
                log(f'skip (complete): {model}/{ptb_type}', log_path)
                continue
            total_est += 1
            run_one(model, ptb_type, args.out_root, args.n_samples, log_path)

    log(f'ALL DONE. {total_est} (model, type) jobs executed.', log_path)


if __name__ == '__main__':
    main()
