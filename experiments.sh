# Base experiments
# python main.py --model 'gpt2' --ptb-type 'char'
# python main.py --model 'gpt2' --ptb-type 'token'
# python main.py --model 'gpt2' --ptb-type 'shuffle'

for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'; do
    for ptb_type in 'char' 'token' 'shuffle'; do
        python main.py --model $model --ptb-type $ptb_type
    done
done


for model in 'qwen2.5_0.5b' 'qwen2.5_1.5b' 'qwen2.5_1.5b' 'qwen2.5_7b' 'qwen2.5_72b' ; do
    for ptb_type in 'char' 'token' 'shuffle'; do
        python main.py --model $model --ptb-type $ptb_type
    done
done

# Ablation experiments - Only run after base
# python ablations.py --ptb-type 'char'
# python ablations.py --ptb-type 'token'
# python ablations.py --ptb-type 'shuffle'