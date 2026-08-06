"""
Compute ALL evaluation metrics.
"""


import torch
import Levenshtein
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform


def eval_loop(inputs_base, outputs_base, inputs_perturb, outputs_perturb, tokenizer, i, output_only=False):
    seq_cols = {
        'sample': [x for x in range(outputs_base.logits.shape[0])],
        'nll': nll(inputs_perturb, outputs_perturb),
        'nll_base': nll(inputs_base, outputs_base),
        'output_divergence': output_divergence(outputs_base, outputs_perturb, tokenizer),
    }
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
        tok_level = pd.DataFrame({
            **get_sample_and_token_indices(inputs_base),
            **activation_similarity(outputs_base, outputs_perturb),
            **linear_cka(outputs_base, outputs_perturb),
            **twoNN_intrinsic_dim(outputs_base, outputs_perturb),
            **mknn_intrinsic_dim(outputs_base, outputs_perturb),
            **attention_entropy(outputs_perturb),
            'logit_kl': logit_kl(outputs_base, outputs_perturb)
        })
    tok_level['sample'] = [x + i*4 for x in tok_level['sample']]
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


def nll(inputs, outputs):
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
    return seq_losses.tolist()


def perplexity(inputs, outputs):
    seq_losses = torch.tensor(nll(inputs, outputs))
    return torch.exp(seq_losses).tolist()


def next_token_accuracy(inputs, outputs):
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    preds = torch.argmax(shift_logits, dim=-1)
    mask = (shift_labels != -100).float()
    correct = ((preds == shift_labels).float() * mask).sum(dim=1)
    return (correct / mask.sum(dim=1)).tolist()


def output_divergence(outputs_base, outputs_perturb, tokenizer):
    text_base_out = tokenizer.batch_decode(torch.argmax(outputs_base.logits[:, :-1, :], dim=-1).cpu())
    text_ptb_out = tokenizer.batch_decode(torch.argmax(outputs_perturb.logits[:, :-1, :], dim=-1).cpu())
    return [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(text_base_out, text_ptb_out)]


def logit_kl(outputs_base, outputs_perturb):
    logits_base = outputs_base.logits[:, :-1, :]
    logits_ptb = outputs_perturb.logits[:, :-1, :]
    log_probs_base = torch.nn.functional.log_softmax(logits_base, dim=-1)
    probs_base = log_probs_base.exp()
    log_probs_ptb = torch.nn.functional.log_softmax(logits_ptb, dim=-1)
    kl = torch.sum(probs_base * (log_probs_base - log_probs_ptb), dim=-1)
    return kl.flatten().tolist()


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
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    num = torch.norm(Y.t() @ X) ** 2
    den = torch.norm(X.t() @ X) * torch.norm(Y.t() @ Y)
    return (num / (den + 1e-9)).item()


DROP_K = 5


def _drop_top_var_dims(X, Y, k):
    if k <= 0:
        return X, Y
    keep = torch.argsort(X.var(dim=0))[:-k]
    return X[:, keep], Y[:, keep]


def linear_cka(outputs_base, outputs_perturb, k=DROP_K):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    n_samples = base_hidden[0].shape[0]
    seq_len = base_hidden[0].shape[1] - 1
    ret = {}
    for L in range(len(base_hidden)):
        cka_vals = []
        for b in range(n_samples):
            X = base_hidden[L][b, :-1, :].float()
            Y = ptb_hidden[L][b, :-1, :].float()
            Xs, Ys = _drop_top_var_dims(X, Y, k)
            cka_vals.append(_linear_cka(Xs, Ys))
        # Broadcast per-sample CKA to per-token positions
        broadcast = []
        for b in range(n_samples):
            broadcast.extend([cka_vals[b]] * seq_len)
        ret[f'cka_layer_{L}'] = broadcast
    return ret


def activation_cka(outputs_base, outputs_perturb, k=DROP_K):
    """Per-sample CKA values (for seq_level). One value per sample."""
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    n_samples = base_hidden[0].shape[0]
    ret = {}
    for L in range(len(base_hidden)):
        cka_vals = []
        for b in range(n_samples):
            X = base_hidden[L][b, :-1, :].float()
            Y = ptb_hidden[L][b, :-1, :].float()
            Xs, Ys = _drop_top_var_dims(X, Y, k)
            cka_vals.append(_linear_cka(Xs, Ys))
        ret[f'activation_cka_layer_{L}'] = cka_vals
    return ret


def twoNN_intrinsic_dim(outputs_base, outputs_perturb, n_samples=500):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    batch_size, seq_len, _ = base_hidden[0].shape
    n_tokens_per_sample = seq_len - 1
    n_total_tokens = batch_size * n_tokens_per_sample
    ret = {}
    for layer_idx in range(len(base_hidden)):
        base_h = base_hidden[layer_idx][:, :-1, :]
        ptb_h = ptb_hidden[layer_idx][:, :-1, :]
        dim_clean = _estimate_intrinsic_dim(base_h, n_samples)
        dim_perturbed = _estimate_intrinsic_dim(ptb_h, n_samples)
        ret[f'intrinsic_dim_clean_layer_{layer_idx}'] = [dim_clean or 0.0] * n_total_tokens
        ret[f'intrinsic_dim_perturbed_layer_{layer_idx}'] = [dim_perturbed or 0.0] * n_total_tokens
        if dim_clean is not None and dim_perturbed is not None:
            change = dim_perturbed - dim_clean
        else:
            change = 0.0
        ret[f'intrinsic_dim_change_layer_{layer_idx}'] = [change] * n_total_tokens
    return ret


def _estimate_intrinsic_dim(hidden_states, n_samples=500):
    """2NN intrinsic dimension from a (n_batch, n_seq, d_model) hidden-state tensor."""
    points = hidden_states.reshape(-1, hidden_states.shape[-1]).float().cpu().numpy()
    return estimate_intrinsic_dim_2nn(points, n_samples=n_samples, n_use=1000, seed=42)


def estimate_intrinsic_dim_2nn(points, n_samples=500, n_use=1000, seed=42):
    """Two-Nearest-Neighbours intrinsic dimension from an (N, D) point matrix.

    Sub-samples down to `n_samples` points (if more are given), then estimates
    the 2NN intrinsic dimension on at most `n_use` points. Returns None when
    there are too few valid points to estimate reliably.
    """
    points = np.asarray(points, dtype=np.float32)
    n_total = points.shape[0]
    if n_total < 10:
        return None
    if n_total > n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=n_samples, replace=False)
        points = points[idx]
        n_total = n_samples
    try:
        n_use = min(n_total, n_use)
        if n_use < n_total:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_total, size=n_use, replace=False)
            points_sub = points[idx]
        else:
            points_sub = points
            n_use = n_total
        dist_matrix = squareform(pdist(points_sub, metric='euclidean'))
        np.fill_diagonal(dist_matrix, np.inf)
        sorted_dists = np.sort(dist_matrix, axis=1)
        r1 = sorted_dists[:, 0]
        r2 = sorted_dists[:, 1]
        valid = (r1 > 1e-10) & (r2 > 1e-10)
        r1 = r1[valid]
        r2 = r2[valid]
        if len(r1) < 10:
            return None
        mu = np.log(r2 / r1)
        intrinsic_dim = len(mu) / np.sum(mu)
        return float(intrinsic_dim)
    except Exception as e:
        return None


def mknn_intrinsic_dim(outputs_base, outputs_perturb, n_samples=500):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    batch_size, seq_len, _ = base_hidden[0].shape
    n_tokens_per_sample = seq_len - 1
    n_total_tokens = batch_size * n_tokens_per_sample
    ret = {}
    for layer_idx in range(len(base_hidden)):
        base_h = base_hidden[layer_idx][:, :-1, :]
        ptb_h = ptb_hidden[layer_idx][:, :-1, :]
        dim_clean = _estimate_mknn_dim(base_h, n_samples)
        dim_perturbed = _estimate_mknn_dim(ptb_h, n_samples)
        ret[f'intrinsic_dim_mknn_clean_layer_{layer_idx}'] = [dim_clean or 0.0] * n_total_tokens
        ret[f'intrinsic_dim_mknn_perturbed_layer_{layer_idx}'] = [dim_perturbed or 0.0] * n_total_tokens
        if dim_clean is not None and dim_perturbed is not None:
            change = dim_perturbed - dim_clean
        else:
            change = 0.0
        ret[f'intrinsic_dim_mknn_change_layer_{layer_idx}'] = [change] * n_total_tokens
    return ret


def _estimate_mknn_dim(hidden_states, n_samples=500):
    """Maximum-Likelihood / MKNN intrinsic dimension from a hidden-state tensor."""
    points = hidden_states.reshape(-1, hidden_states.shape[-1]).float().cpu().numpy()
    return estimate_intrinsic_dim_mknn(points, n_samples=n_samples, n_use=1000, seed=42)


def estimate_intrinsic_dim_mknn(points, n_samples=500, n_use=1000, seed=42):
    """Maximum-likelihood intrinsic dimension (MKNN) from an (N, D) point matrix."""
    points = np.asarray(points, dtype=np.float32)
    n_total = points.shape[0]
    if n_total < 10:
        return None
    if n_total > n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_total, size=n_samples, replace=False)
        points = points[idx]
        n_total = n_samples
    try:
        n_use = min(n_total, n_use)
        if n_use < n_total:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_total, size=n_use, replace=False)
            points_sub = points[idx]
        else:
            points_sub = points
            n_use = n_total
        dist_matrix = squareform(pdist(points_sub, metric='euclidean'))
        np.fill_diagonal(dist_matrix, np.inf)
        sorted_dists = np.sort(dist_matrix, axis=1)
        r1 = sorted_dists[:, 0]
        valid = r1 > 1e-10
        r1 = r1[valid]
        if len(r1) < 10:
            return None
        r_min = np.min(r1)
        intrinsic_dim = len(r1) / np.sum(np.log(r1 / r_min))
        return float(intrinsic_dim)
    except Exception as e:
        return None


def attention_entropy(outputs):
    attentions = outputs.attentions
    _, nh, _, _ = attentions[0].shape
    ret = {}
    for i in range(len(attentions)):
        for h in range(nh):
            head_att = attentions[i][:, h, :-1, :]
            seq_len = head_att.shape[-1]
            mask = head_att > 0
            safe_att = head_att.clamp(min=1e-9)
            ent = -torch.sum(mask * head_att * torch.log(safe_att), dim=-1)
            max_ent = torch.log(torch.tensor(float(seq_len)))
            ret[f'attn_layer{i}_head_{h}_entropy_norm'] = (ent / max_ent).flatten().tolist()
    return ret