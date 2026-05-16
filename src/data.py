from datasets import load_dataset

import string


def load_eval_dataset(rng, n_samples=None, perturb_pct = 0, ptb_type="char", tokenizer=None, model=None):
    """
    Legal perturbation types: ['char', 'token', 'shuffle']
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    texts = list(ds['test']['text'])
    texts = [x for x in texts if len(x.split()) > 20]
    rng.shuffle(texts)
    if n_samples:
        texts = texts[:n_samples]
    
    if ptb_type == 'char':
        sub_pool = string.ascii_letters + string.digits + string.punctuation
    elif ptb_type == 'token':
        sub_pool = tokenizer.get_vocab()

    # TODO: Returning in list format is a hack. Figure out how to implement perturbations and keep hf dataset
    return perturb(texts, perturb_pct, rng, ptb_type, sub_pool)


def perturb(texts, perturb_pct, rng, ptb_type, sub_pool):
    ret = []
    for text in texts:
        ret.append(perturb_text(text, rng, ptb_type, perturb_pct, sub_pool))
    return ret


# ptb_type = noise, shuffling, remapping
def perturb_text(text, rng, ptb_type, perturb_pct, sub_pool):
    if ptb_type == 'char':
        ret = []
        for c in text:
            if rng.randint(1, 100) < perturb_pct:
                ret.append(rng.choice(sub_pool))
            else:
                ret.append(c)
        return ''.join(ret)
    elif ptb_type == 'token':
        if rng.randint(1, 100) < perturb_pct:
            return rng.choice(sub_pool)
        return ret
    elif ptb_type == 'shuffle':
        return ret
    assert False, "PTB type must be one of ['char', 'token', 'shuffle']"