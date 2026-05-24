# Base experiments
# for i in $(seq 0 5 50); do
#     python main.py --ptb-type 'char' --ptb-pct $i
# done

# for i in $(seq 5 5 50); do
#     python main.py --ptb-type 'token' --ptb-pct $i
# done

# for i in $(seq 0 5 100); do
#     python main.py --ptb-type 'shuffle' --ptb-pct $i
# done

# Ablation experiments - Only run after base
python ablations.py --ptb-type 'char'
python ablations.py --ptb-type 'token'
python ablations.py --ptb-type 'shuffle'