# Optionally set HF_HOME to control where model/dataset downloads are cached:
#   export HF_HOME=/path/to/hf_cache
#
# GPU to run on (override with: CUDA_VISIBLE_DEVICES=2 bash experiments.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Base experiments
# python main.py --model 'gpt2' --ptb-type 'char'
# python main.py --model 'gpt2' --ptb-type 'token'
# python main.py --model 'gpt2' --ptb-type 'shuffle'

for model in 'gpt2' do #'gpt2-medium' 'gpt2-large' 'gpt2-xl'; do
    for ptb_type in 'char' 'token' 'shuffle'; do
        for last_n_tok in 1 5 10 25 50; do
            python main.py --model $model --ptb-type $ptb_type --num-eval-tokens $last_n_tok
        done
    done
done


# for model in 'qwen2.5_0.5b' 'qwen2.5_1.5b' 'qwen2.5_1.5b' 'qwen2.5_7b' 'qwen2.5_72b' ; do
# for model in 'qwen2.5_14b' 'qwen2.5_32b' 'qwen2.5_72b' ; do
#     for ptb_type in 'char' 'token' 'shuffle'; do
#         python main.py --model $model --ptb-type $ptb_type
#     done
# done

# Ablation experiments - Only run after base
# python ablations.py --ptb-type 'char'
# python ablations.py --ptb-type 'token'
# python ablations.py --ptb-type 'shuffle'