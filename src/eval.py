"""
Compute ALL evaluation metrics.
"""


import torch
import Levenshtein
import pandas as pd
from scipy.stats import entropy


def eval_loop(inputs_base, outputs_base, inputs_perturb, outputs_perturb, tokenizer, i, output_only=False, num_eval_tokens=0):
    seq_cols = {
        'sample': [x for x in range(outputs_base.logits.shape[0])],
        'nll': nll(inputs_perturb, outputs_perturb, num_eval_tokens=num_eval_tokens),
        'output_divergence': output_divergence(outputs_base, outputs_perturb, tokenizer, num_eval_tokens=num_eval_tokens),
    }
    # Per-sample linear CKA: a representation-similarity metric robust to the
    # massive-activation outlier dims that inflate raw cosine at late layers.
    if not output_only:
        seq_cols.update(activation_cka(outputs_base, outputs_perturb, num_eval_tokens=num_eval_tokens))
    seq_level = pd.DataFrame(seq_cols)

    seq_level['sample'] = [x+i*4 for x in seq_level['sample']]

    # TODO: Logit KL on only the last n ptbs
    if output_only:
        tok_level = pd.DataFrame({
            **get_sample_and_token_indices(inputs_base),
            'logit_kl': logit_kl(outputs_base, outputs_perturb, num_eval_tokens=num_eval_tokens)
        })
    else:
        tok_level = pd.DataFrame({
            **get_sample_and_token_indices(inputs_base),
            **activation_similarity(outputs_base, outputs_perturb, num_eval_tokens=num_eval_tokens),
            **attention_entropy(outputs_perturb, num_eval_tokens=num_eval_tokens),
            'logit_kl': logit_kl(outputs_base, outputs_perturb, num_eval_tokens=num_eval_tokens),
        })
    tok_level['sample'] = [x+i*4 for x in tok_level['sample']]
    tok_level = tok_level.groupby('sample',as_index=False).mean()
    return pd.merge(seq_level, tok_level, on='sample', how='inner')

def get_sample_and_token_indices(inputs_base):
    n_samples, sample_length = inputs_base.input_ids.shape

    sample_length -= 1

    sample_idx = []
    token_idx = []

    for i in range(n_samples):
        sample_idx.extend([i]*sample_length),
        token_idx.extend([x for x in range(sample_length)])

    return {
        'sample': sample_idx,
        'token_in_sample': token_idx
    }

def nll(inputs, outputs, num_eval_tokens=0):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    if num_eval_tokens > 0:
        shift_labels = shift_labels[:, :-num_eval_tokens]
        shift_logits = shift_logits[:, :-num_eval_tokens, :]
    
    token_losses = torch.nn.CrossEntropyLoss(reduction='none')(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(input_ids.size(0), -1) 
    
    mask = (shift_labels != -100).float()
    seq_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
    # seq_perplexities = torch.exp(seq_losses)

    return seq_losses.tolist()

def output_divergence(outputs_base, outputs_perturb, tokenizer, num_eval_tokens=0):
    text_base_out = tokenizer.batch_decode(torch.argmax(outputs_base.logits[:, :-1, :], dim=-1).cpu())
    text_ptb_out = tokenizer.batch_decode(torch.argmax(outputs_perturb.logits[:, :-1, :], dim=-1).cpu())

    if num_eval_tokens > 0:
        text_base_out = [x[:-num_eval_tokens] for x in text_base_out]
        text_ptb_out = [x[:-num_eval_tokens] for x in text_ptb_out]

    return [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(text_base_out, text_ptb_out)]

# Make robust
def logit_kl(outputs_base, outputs_perturb, num_eval_tokens=0):
    logits_base = outputs_base.logits[:, :-1, :]
    logits_ptb = outputs_perturb.logits[:, :-1, :]
    if num_eval_tokens > 0:
        logits_base = logits_base[:, :-num_eval_tokens, :]
        logits_ptb = logits_ptb[:, :-num_eval_tokens, :]

    log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)
    probs_base = log_probs_base.exp()
    log_probs_ptb = torch.nn.functional.log_softmax(logits_ptb, dim=-1)

    kl = torch.sum(probs_base * (log_probs_base - log_probs_ptb), dim=-1)
    return kl.flatten().tolist()

def topk_divergence(outputs_base, outputs_perturb, k=50):
    cutoffs_base = torch.min(torch.topk(outputs_base.logits[:, :-1, :], k, dim=-1).values, dim=-1, keepdims=True).values
    base_intopk = torch.where(
        outputs_base.logits[:, :-1, :] >= cutoffs_base,
        torch.ones_like(outputs_base.logits[:, :-1, :]), 
        torch.zeros_like(outputs_base.logits[:, :-1, :])
    )


    cutoffs_ptb = torch.min(torch.topk(outputs_perturb.logits[:, :-1, :], k, dim=-1).values, dim=-1, keepdims=True).values
    ptb_intopk = torch.where(
        outputs_perturb.logits[:, :-1, :] >= cutoffs_ptb,
        torch.ones_like(outputs_perturb.logits[:, :-1, :]), 
        torch.zeros_like(outputs_perturb.logits[:, :-1, :])
    )

    overlap_topk = torch.sum(base_intopk * ptb_intopk, dim=-1) / k

    return overlap_topk.flatten().tolist()

# make robust
def activation_similarity(outputs_base, outputs_perturb, num_eval_tokens=0):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states

    ret = {}

    for i, _ in enumerate(zip(base_hidden, ptb_hidden)):
        base_i = base_hidden[i][:, :-1, :]
        ptb_i = ptb_hidden[i][:, :-1, :]

        if num_eval_tokens > 0:
            base_i = base_i[:, :-num_eval_tokens, :]
            ptb_i = ptb_i[:, :-num_eval_tokens, :]

        cos_sim = torch.cosine_similarity(base_i, ptb_i, dim=-1).clamp(-1, 1)
        ret[f'activation_cos_sim_layer_{i}'] = cos_sim.flatten().tolist()

        l2 = torch.sum((base_i - ptb_i) ** 2, dim=-1)
        ret[f'activation_l2_dist_layer_{i}'] = l2.flatten().tolist()

    return ret

def _linear_cka(X, Y):
    """
    Linear CKA between two (n_tokens, d) activation matrices.
    Centers each feature (removes the massive-activation constant offset) but
    does NOT rescale per-dimension, so near-constant noise dims stay negligible.
    Returns a scalar in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = torch.norm(Y.t() @ X) ** 2
    den = torch.norm(X.t() @ X) * torch.norm(Y.t() @ Y)
    return (num / (den + 1e-9)).item()

DROP_K = 5  # drop the k highest-variance (massive-activation) dims per layer

def _drop_top_var_dims(X, Y, k):
    """Drop the k dims with largest across-token variance (measured on clean X)."""
    if k <= 0:
        return X, Y
    keep = torch.argsort(X.var(dim=0))[:-k]
    return X[:, keep], Y[:, keep]

def activation_cka(outputs_base, outputs_perturb, k=DROP_K, num_eval_tokens=0):
    """
    Per-sample representation similarity AFTER dropping the top-k highest-variance
    dims per layer. Those outlier dims (massive activations) confound both raw
    cosine (via magnitude) and plain CKA (via variance); removing them leaves the
    genuine content signal, on which stripped cosine and stripped CKA agree.
    """
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    n_samples = base_hidden[0].shape[0]

    ret = {}
    for L in range(len(base_hidden)):
        cka_vals, cos_vals = [], []
        for b in range(n_samples):
            X = base_hidden[L][b, :-1, :].float()
            Y = ptb_hidden[L][b, :-1, :].float()
            if num_eval_tokens > 0:
                X = X[:-num_eval_tokens, :]
                Y = Y[:-num_eval_tokens, :]
            Xs, Ys = _drop_top_var_dims(X, Y, k)
            cka_vals.append(_linear_cka(Xs, Ys))
            cos_vals.append(torch.cosine_similarity(Xs, Ys, dim=-1).mean().item())
        ret[f'activation_cka_layer_{L}'] = cka_vals
        ret[f'activation_cos_stripped_layer_{L}'] = cos_vals
    return ret

def attention_entropy(outputs, num_eval_tokens=0):
    attentions = outputs.attentions
    _, nh, _, _ = attentions[0].shape

    ret = {}
    for i in range(len(attentions)):
        for h in range(nh):
            head_att = attentions[i][:, h, :-1, :]
            if num_eval_tokens > 0:
                head_att = head_att[:, :-num_eval_tokens, :]
            seq_len = head_att.shape[-1]
            mask = head_att > 0
            safe_att = head_att.clamp(min=1e-9)
            ent = -torch.sum(mask * head_att * torch.log(safe_att), dim=-1)
            max_ent = torch.log(torch.tensor(float(seq_len)))
            ret[f'attn_layer{i}_head_{h}_entropy_norm'] = (ent / max_ent).flatten().tolist()
    return ret