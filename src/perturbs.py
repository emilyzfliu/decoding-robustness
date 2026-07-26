"""
File for all perturbation implementations.
Outside code should only ever call function `perturb`
"""
import string
from copy import deepcopy

def perturb(texts, perturb_pct, rng, ptb_type, tokenizer, num_eval_tokens=0):
    if ptb_type == 'char':
        return character_substitution(texts, perturb_pct, rng, num_eval_tokens)
    elif ptb_type == 'token':
        return token_substitution(texts, perturb_pct, rng, tokenizer, num_eval_tokens)
    elif ptb_type == 'shuffle':
        return token_shuffle(texts, perturb_pct, rng, num_eval_tokens)
    else:
        raise TypeError("ptb_type must be one of ['char', 'token', 'shuffle']")
    

def character_substitution(texts, perturb_pct, rng, num_eval_tokens=0):
    """
    Substitute random ASCII characters in the text input.
    """
    sub_pool = string.ascii_letters + string.digits + string.punctuation
    ret = []
    for text in texts:
        word = []
        if num_eval_tokens > 0:
            orig = text
            text = ' '.join(orig.split()[:-num_eval_tokens])
            holdout = ' '.join(orig.split()[-num_eval_tokens:])
        for c in text:
            if rng.randint(1, 100) < perturb_pct:
                word.append(rng.choice(sub_pool))
            else:
                word.append(c)
        if num_eval_tokens > 0:
            word.append(' ')
            word.append(holdout)
        ret.append(''.join(word))
    return ret

def token_substitution(texts, perturb_pct, rng, tokenizer, max_length=128, num_eval_tokens=0):
    """
    Substitute random tokens from the tokenizer vocabulary.
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )
    
    ret = []
    for input_ids in encodings['input_ids']:
        if num_eval_tokens > 0:
            orig_ids = deepcopy(input_ids)
            input_ids = orig_ids[:-num_eval_tokens]
            holdout = orig_ids[-num_eval_tokens:]
        
        new_ids = [
            rng.randint(0, tokenizer.vocab_size - 1) 
            if rng.randint(1, 100) < perturb_pct 
            else tid
            for tid in input_ids
        ]

        if num_eval_tokens > 0:
            new_ids = new_ids + holdout
        ret.append(tokenizer.decode(new_ids))
    
    return ret


def token_shuffle(texts, perturb_pct, rng, num_eval_tokens=0):
    """
    Scramble around words in the text.
    """
    ret = []
    for text in texts:
        toks = text.split()
        if num_eval_tokens > 0:
            orig = deepcopy(toks)
            toks = orig[:-num_eval_tokens]
            holdout = orig[-num_eval_tokens:]
        shuffle_window = int(perturb_pct*len(toks) / 100)

        start = rng.randint(0, len(toks) - shuffle_window - 1) if perturb_pct < 100 else 0

        tok_to_shuffle = toks[start:start+shuffle_window]
        rng.shuffle(tok_to_shuffle)

        new_toks = toks[:start] + tok_to_shuffle + toks[start+shuffle_window:]
        if num_eval_tokens > 0:
            new_toks = new_toks + holdout
        ret.append(' '.join(new_toks))
    return ret
