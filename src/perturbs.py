"""
File for all perturbation implementations.
Outside code should only ever call function `perturb`
"""

def perturb(texts, perturb_pct, rng, ptb_type, sub_pool):
    if ptb_type == 'char':
        return character_substitution(texts, perturb_pct, rng, sub_pool)
    elif ptb_type == 'token':
        return token_substitution(texts, perturb_pct, rng, sub_pool)
    elif ptb_type == 'shuffle':
        return token_shuffle(texts, perturb_pct, rng, sub_pool)
    elif ptb_type == 'noise':
        return token_insertion(texts, perturb_pct, rng, sub_pool)
    else:
        raise TypeError("ptb_type must be one of ['char', 'token', 'shuffle']")
    

def character_substitution(texts, perturb_pct, rng, sub_pool):
    """
    Substitute random ASCII characters in the text input.
    """
    ret = []
    for text in texts:
        word = []
        for c in text:
            if rng.randint(1, 100) < perturb_pct:
                word.append(rng.choice(sub_pool))
            else:
                word.append(c)
        ret.append(''.join(word))
    return ret

def token_substitution(texts, perturb_pct, rng, sub_pool):
    """
    Substitute random tokens from the tokenizer vocabulary.
    """
    ret = []
    for text in texts:
        if rng.randint(1, 100) < perturb_pct:
            ret.append(rng.choice(sub_pool))
        else:
            ret.append(text)
    return ret


def token_shuffle(texts, perturb_pct, rng, sub_pool):
    """
    Substitute random ASCII characters in the text.
    """
    ret = []
    for text in texts:
        toks = text.split()
        shuffle_window = int(perturb_pct*len(toks) / 100)

        start = rng.randint(0, len(toks) - shuffle_window - 1)

        tok_to_shuffle = toks[start:start+shuffle_window]
        rng.shuffle(tok_to_shuffle)

        new_toks = toks[:start] + tok_to_shuffle + toks[start+shuffle_window:]
        ret.append(' '.join(new_toks))
    return ret


def token_insertion(texts, perturb_pct, rng, sub_pool):
    ret = []
    for text in texts:
        if rng.randint(1, 100) < perturb_pct:
            ret.append(rng.choice(sub_pool))
        ret.append(text)
    return ret