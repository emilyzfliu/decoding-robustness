# Optionally set HF_HOME to control where model/dataset downloads are cached:
#   export HF_HOME=/path/to/hf_cache
#
# GPU to run on (override with: CUDA_VISIBLE_DEVICES=2 bash experiments.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Base experiments
for i in $(seq 0 5 50); do
    python main.py --ptb-type 'char' --ptb-pct $i
done

for i in $(seq 5 5 50); do
    python main.py --ptb-type 'token' --ptb-pct $i
done

for i in $(seq 0 5 100); do
    python main.py --ptb-type 'shuffle' --ptb-pct $i
done

# Ablation experiments - Only run after base
python ablations.py --ptb-type 'char'
python ablations.py --ptb-type 'token'
python ablations.py --ptb-type 'shuffle'