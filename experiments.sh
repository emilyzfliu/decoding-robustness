# Base experiments
python main.py --model 'gpt2' --ptb-type 'char'
python main.py --model 'gpt2' --ptb-type 'token'
python main.py --model 'gpt2' --ptb-type 'shuffle'
# Ablation experiments - Only run after base
# python ablations.py --ptb-type 'char'
# python ablations.py --ptb-type 'token'
# python ablations.py --ptb-type 'shuffle'