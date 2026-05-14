from datasets import load_dataset

def load_eval_dataset(n_samples, rng, perturb_pct = 0, ptb_type="noise"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    # TODO: Clean the dataset to ensure high quality samples.
    texts = list(ds['test']['text'])
    rng.shuffle(texts)
    texts = texts[:n_samples]

    # TODO: Returning in list format is a hack. Figure out how to implement perturbations and keep hf dataset
    return perturb(texts, perturb_pct, rng, ptb_type)


def perturb(texts, perturb_pct, rng, ptb_type):
    ret = []
    for text in texts:
        if rng.randint(1, 100) < perturb_pct:
            ret.append(perturb_text(text, ptb_type))
        else:
            ret.append(text)
    return ret


# ptb_type = noise, shuffling, remapping
def perturb_text(text, ptb_type):
    # TODO
    pass