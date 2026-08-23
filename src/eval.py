"""
Compute ALL evaluation metrics.
"""

import torch
import Levenshtein
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform


# Computes core evaluation metrics only for perturbation-trend and cross model experiments.
def eval_loop(outputs_base, inputs_perturb, outputs_perturb, tokenizer, i, output_only=False):
    seq_cols = {
        'sample': [x for x in range(outputs_base.logits.shape[0])],
        'nll': nll(inputs_perturb, outputs_perturb),
        'output_divergence': output_divergence(outputs_base, outputs_perturb, tokenizer),
        'last_token_kl': logit_kl(outputs_base, outputs_perturb)
    }
    if not output_only:
        seq_cols.update(activation_cka(outputs_base, outputs_perturb))
        seq_cols.update(intrinsic_dims(outputs_base, outputs_perturb))

    res = pd.DataFrame(seq_cols)
    return res

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


def output_divergence(outputs_base, outputs_perturb, tokenizer):
    text_base_out = tokenizer.batch_decode(torch.argmax(outputs_base.logits[:, :-1, :], dim=-1).cpu())
    text_ptb_out = tokenizer.batch_decode(torch.argmax(outputs_perturb.logits[:, :-1, :], dim=-1).cpu())
    return [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(text_base_out, text_ptb_out)]


def logit_kl(outputs_base, outputs_perturb):
    logits_base = outputs_base.logits[:, -1, :]
    logits_ptb = outputs_perturb.logits[:, -1, :]
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
    """TwoNN intrinsic dimensions (2NN block of intrinsic_dims).

    Kept for backwards compatibility with analysis scripts; the eval loop calls
    ``intrinsic_dims`` once so both estimators share one distance matrix.
    """
    return {k: v for k, v in intrinsic_dims(outputs_base, outputs_perturb, n_samples=n_samples).items()
            if not k.startswith('intrinsic_dim_mknn')}


def mknn_intrinsic_dim(outputs_base, outputs_perturb, n_samples=500):
    """MKNN intrinsic dimensions (MKNN block of intrinsic_dims).

    Kept for backwards compatibility with analysis scripts; the eval loop calls
    ``intrinsic_dims`` once so both estimators share one distance matrix.
    """
    return {k: v for k, v in intrinsic_dims(outputs_base, outputs_perturb, n_samples=n_samples).items()
            if k.startswith('intrinsic_dim_mknn')}


def intrinsic_dims(outputs_base, outputs_perturb, n_samples=500):
    """TwoNN and MKNN intrinsic dims (clean/perturbed/change per layer).

    Both estimators share the same subsampled distance matrix per layer, so
    the expensive O(n^2) pairwise-distance computation is done once per layer
    instead of twice (one per estimator).
    """
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states
    batch_size, seq_len, _ = base_hidden[0].shape
    # n_tokens_per_sample = seq_len - 1
    # n_total_tokens = batch_size * n_tokens_per_sample
    est = []
    for layer_idx in range(len(base_hidden)):
        base_h = base_hidden[layer_idx][:, :-1, :]
        ptb_h = ptb_hidden[layer_idx][:, :-1, :]
        clean_2nn, clean_mknn = _estimate_dims_from_points(base_h, n_samples)
        ptb_2nn, ptb_mknn = _estimate_dims_from_points(ptb_h, n_samples)
        est.append((layer_idx, clean_2nn, ptb_2nn, clean_mknn, ptb_mknn))
    ret = {}
    for layer_idx, clean_2nn, ptb_2nn, _, _ in est:
        ret[f'intrinsic_dim_clean_layer_{layer_idx}'] = [clean_2nn or 0.0] * batch_size
        ret[f'intrinsic_dim_perturbed_layer_{layer_idx}'] = [ptb_2nn or 0.0] * batch_size
        if clean_2nn is not None and ptb_2nn is not None:
            change_2nn = ptb_2nn - clean_2nn
        else:
            change_2nn = 0.0
        ret[f'intrinsic_dim_change_layer_{layer_idx}'] = [change_2nn]
    for layer_idx, _, _, clean_mknn, ptb_mknn in est:
        ret[f'intrinsic_dim_mknn_clean_layer_{layer_idx}'] = [clean_mknn or 0.0] * batch_size
        ret[f'intrinsic_dim_mknn_perturbed_layer_{layer_idx}'] = [ptb_mknn or 0.0] * batch_size
        if clean_mknn is not None and ptb_mknn is not None:
            change_mknn = ptb_mknn - clean_mknn
        else:
            change_mknn = 0.0
        ret[f'intrinsic_dim_mknn_change_layer_{layer_idx}'] = [change_mknn] * batch_size
    return ret


def _estimate_dims_from_points(hidden_states, n_samples=500):
    """(TwoNN, MKNN) intrinsic dims from a (n_batch, n_seq, d_model) tensor,
    computing one shared distance matrix for both estimators."""
    points = hidden_states.reshape(-1, hidden_states.shape[-1]).float()
    return _estimate_dims_2nn_mknn(points, n_samples)


def _estimate_dims_2nn_mknn(points, n_samples=500, seed=42):
    """
    (TwoNN, MKNN) intrinsic dimensions from an (N, D) point matrix.

    Sub-samples down to ``n_samples`` points, computes the distance matrix
    once, and derives both estimators from the same sorted neighbour
    distances. Returns (None, None) when the estimate is degenerate.

    points: (N, D) tensor on GPU, already detached and float32.
    """
    n_total = points.shape[0]
    if n_total < 10:
        return None, None

    if n_total > n_samples:
        g = torch.Generator(device=points.device).manual_seed(seed)
        rand_indices = torch.randperm(n_total, generator=g, device=points.device)[:n_samples]
        points = points[rand_indices]

    with torch.no_grad():
        dists = torch.cdist(points, points)
        torch.diagonal(dists).fill_(float('inf'))

        values, _ = torch.topk(dists, k=2, largest=False, dim=1)
        r1 = values[:, 0]
        r2 = values[:, 1]

        valid = (r1 > 1e-10) & (r2 > 1e-10)
        r1_v, r2_v = r1[valid], r2[valid]
        dim_2nn = None
        if len(r1_v) >= 10:
            mu = torch.log(r2_v / r1_v)
            denom = torch.sum(mu)
            if torch.isfinite(denom) and denom > 1e-10:
                dim_2nn = float(len(mu) / denom)

        valid_mknn = r1 > 1e-10
        r1_m = r1[valid_mknn]
        dim_mknn = None
        if len(r1_m) >= 10:
            r_min = torch.min(r1_m)
            denom = torch.sum(torch.log(r1_m / r_min))
            if torch.isfinite(denom) and denom > 1e-10:
                dim_mknn = float(len(r1_m) / denom)

    return dim_2nn, dim_mknn

def estimate_intrinsic_dim_2nn(points, n_samples=500, n_use=1000, seed=42):
    """Two-Nearest-Neighbours intrinsic dimension from an (N, D) point matrix.

    Sub-samples down to `n_samples` points (if more are given), then estimates
    the 2NN intrinsic dimension on at most `n_use` points. Returns None when
    there are too few valid points or the estimate is degenerate.
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
        denom = np.sum(mu)
        if not np.isfinite(denom) or denom <= 1e-10:
            return None
        return float(len(mu) / denom)
    except Exception as e:
        return None


def estimate_intrinsic_dim_mknn(points, n_samples=500, n_use=1000, seed=42):
    """Maximum-likelihood intrinsic dimension (MKNN) from an (N, D) point matrix.

    Sub-samples down to `n_samples` points (if more are given), then estimates
    the MKNN intrinsic dimension on at most `n_use` points. Returns None when
    there are too few valid points or the estimate is degenerate.
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
        valid = r1 > 1e-10
        r1 = r1[valid]
        if len(r1) < 10:
            return None
        r_min = np.min(r1)
        denom = np.sum(np.log(r1 / r_min))
        if not np.isfinite(denom) or denom <= 1e-10:
            return None
        return float(len(r1) / denom)
    except Exception as e:
        return None

