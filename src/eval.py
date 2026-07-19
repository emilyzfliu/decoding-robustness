"""
Compute ALL evaluation metrics.
"""


import torch
import Levenshtein
import pandas as pd


def eval_loop(inputs_base, outputs_base, inputs_perturb, outputs_perturb, tokenizer, i, output_only=False):
    seq_cols = {
        'sample': [x for x in range(outputs_base.logits.shape[0])],
        'perplexity': perplexity(inputs_perturb, outputs_perturb),
        'perplexity_baseline': perplexity(inputs_base, outputs_base),
        'output_divergence': output_divergence(outputs_base, outputs_perturb, tokenizer),
    }
    # Per-sample linear CKA: a representation-similarity metric robust to the
    # massive-activation outlier dims that inflate raw cosine at late layers.
    if not output_only:
        seq_cols.update(activation_cka(outputs_base, outputs_perturb))
    seq_level = pd.DataFrame(seq_cols)

    seq_level['sample'] = [x+i*4 for x in seq_level['sample']]
    if output_only:
        tok_level = pd.DataFrame({
            **get_sample_and_token_indices(inputs_base),
            'logit_kl': logit_kl(outputs_base, outputs_perturb)
        })
    else:
        tok_cols = {
            **get_sample_and_token_indices(inputs_base),
            **activation_similarity(outputs_base, outputs_perturb),
            'logit_kl': logit_kl(outputs_base, outputs_perturb),
        }
        # attention entropy only when attentions were computed (skipped for speed
        # /memory when output_attentions=False, e.g. Llama Phase 1)
        if outputs_perturb.attentions is not None:
            tok_cols.update(attention_entropy(outputs_perturb))
        tok_level = pd.DataFrame(tok_cols)
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

def perplexity(inputs, outputs):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    token_losses = torch.nn.CrossEntropyLoss(reduction='none')(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(input_ids.size(0), -1) 
    
    mask = (shift_labels != -100).float()
    seq_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
    seq_perplexities = torch.exp(seq_losses)

    return seq_perplexities.tolist()

def output_divergence(outputs_base, outputs_perturb, tokenizer):
    text_base_out = tokenizer.batch_decode(torch.argmax(outputs_base.logits[:, :-1, :], dim=-1).cpu(), clean_up_tokenization_spaces=False)
    text_ptb_out = tokenizer.batch_decode(torch.argmax(outputs_perturb.logits[:, :-1, :], dim=-1).cpu(), clean_up_tokenization_spaces=False)

    return [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(text_base_out, text_ptb_out)]

def logit_kl(outputs_base, outputs_perturb):
    # per-token KL(P_base || P_ptb); upcast to fp32 for numerical stability
    lp_base = torch.log_softmax(outputs_base.logits[:, :-1, :].float(), dim=-1)
    lp_ptb  = torch.log_softmax(outputs_perturb.logits[:, :-1, :].float(), dim=-1)
    kl = (lp_base.exp() * (lp_base - lp_ptb)).sum(dim=-1)   # (batch, seq-1)
    return kl.flatten().tolist()                            # per-token; only batch*(seq-1) values to CPU

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

def activation_similarity(outputs_base, outputs_perturb):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states

    ret = {}

    for i, _ in enumerate(zip(base_hidden, ptb_hidden)):
        base_i = base_hidden[i][:, :-1, :]
        ptb_i = ptb_hidden[i][:, :-1, :]

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

def activation_cka(outputs_base, outputs_perturb, k=DROP_K):
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
            Xs, Ys = _drop_top_var_dims(X, Y, k)
            cka_vals.append(_linear_cka(Xs, Ys))
            cos_vals.append(torch.cosine_similarity(Xs, Ys, dim=-1).mean().item())
        ret[f'activation_cka_layer_{L}'] = cka_vals
        ret[f'activation_cos_stripped_layer_{L}'] = cos_vals
    return ret

def attention_entropy(outputs):
    attentions = outputs.attentions
    _, nh, _, _ = attentions[0].shape

    ret = {}

    for i in range(len(attentions)):
        for h in range(nh):
            head_att = attentions[i][:,h,:-1,:] # bs, num_tok, num_tok
            mask = head_att > 0
            safe_att = head_att.clamp(min=1e-9)
            entropy = -torch.sum(mask * head_att * torch.log(safe_att), dim=-1)
            
            ret[f'attn_layer{i}_head_{h}_entropy'] = entropy.flatten().tolist()
    
    return ret