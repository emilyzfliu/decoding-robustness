export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}


# for ptb_type in 'char' 'token' 'shuffle' 'word' 'synonym' 'typo' 'adv'; do
#     python analysis/attentions.py --model 'gpt2' --ptb-type $ptb_type --ptb-pct 5 --out-root res_attentions_0820_debug --debug
#     python analysis/attentions.py --model 'gpt2' --ptb-type $ptb_type --ptb-pct 30 --out-root res_attentions_0820_debug --debug
# done


# after verified
for ptb_type in 'word' 'synonym' ; do
    python analysis/attentions.py --model 'gpt2' --ptb-type $ptb_type --ptb-pct 5 --out-root res_attentions_0826
    python analysis/attentions.py --model 'gpt2' --ptb-type $ptb_type --ptb-pct 30 --out-root res_attentions_0826
done

