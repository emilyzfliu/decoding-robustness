# Optionally set HF_HOME to control where model/dataset downloads are cached:
#   export HF_HOME=/path/to/hf_cache
#
# GPU to run on (override with: CUDA_VISIBLE_DEVICES=2 bash experiments.sh)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Base experiments
# python main.py --model 'gpt2' --ptb-type 'char'   --n-samples 100 --batch-size 8
# python main.py --model 'gpt2' --ptb-type 'token'  --n-samples 100 --batch-size 8
# python main.py --model 'gpt2' --ptb-type 'shuffle' --n-samples 100 --batch-size 8
# python main.py --model 'gpt2' --ptb-type 'typo'   --n-samples 100 --batch-size 8
# python main.py --model 'gpt2' --ptb-type 'word'   --n-samples 100 --batch-size 8
# python main.py --model 'gpt2' --ptb-type 'synonym' --n-samples 100 --batch-size 8

# Analysis
# python compare_perturbation_types.py --model gpt2 --pct 25
# python validate_intrinsic_dim.py --model gpt2 --n-seq 8 --ptb-type char --ptb-pct 25
# python figures.py --model gpt2

# for model in 'gpt2' 'gpt2-medium' 'gpt2-large' 'gpt2-xl'; do
#     for ptb_type in 'char' 'token' 'shuffle'; do
#         python main.py --model $model --ptb-type $ptb_type
#     done
# done


# for model in 'qwen2.5_0.5b' 'qwen2.5_1.5b' 'qwen2.5_1.5b' 'qwen2.5_7b' 'qwen2.5_72b' ; do
# for model in 'qwen2.5_14b' 'qwen2.5_32b' 'qwen2.5_72b' ; do
#     for ptb_type in 'char' 'token' 'shuffle'; do
#         python main.py --model $model --ptb-type $ptb_type
#     done
# done


for ptb_type in 'char' 'token' 'shuffle' 'word' 'typo' 'synonym'; do
    # python main.py --ptb-type $ptb_type --ptb-pct 5 --output-tag 'clean_run'
    # python main.py --ptb-type $ptb_type --ptb-pct 30 --output-tag 'clean_run'

    python ablations.py --ptb-type $ptb_type --output-tag 'ablated_run'
done

# Ablation experiments - Only run after base
# python ablations.py --ptb-type 'char'
# python ablations.py --ptb-type 'token'
# python ablations.py --ptb-type 'shuffle'