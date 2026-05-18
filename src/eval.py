"""
Compute ALL evaluation metrics.
"""


import torch
import Levenshtein

def eval_loop(inputs_base, outputs_base, inputs_perturb, outputs_perturb, tokenizer, model):
    print('hidden states',len(outputs_base.hidden_states), outputs_base.hidden_states[0].shape)
    return {
        'base_perplexity': perplexity(inputs_base, outputs_base),
        'ptb_perplexity': perplexity(inputs_perturb, outputs_perturb),
        'output_divergence': output_divergence(outputs_base, outputs_perturb, tokenizer),
        'top5_divergence': topk_divergence(outputs_base, outputs_perturb, 5),
        'top25_divergence': topk_divergence(outputs_base, outputs_perturb, 25),
        'top50_divergence': topk_divergence(outputs_base, outputs_perturb, 50),
        'top100_divergence': topk_divergence(outputs_base, outputs_perturb, 100),
        **activation_similarity(outputs_base, outputs_perturb)
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

    return seq_perplexities

def output_divergence(outputs_base, outputs_perturb, tokenizer):
    text_base_out = tokenizer.batch_decode(torch.argmax(outputs_base.logits[:, :-1, :], dim=-1))
    text_ptb_out = tokenizer.batch_decode(torch.argmax(outputs_perturb.logits[:, :-1, :], dim=-1))

    return [Levenshtein.distance(x, y)/max(len(x), len(y)) for x, y in zip(text_base_out, text_ptb_out)]

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

    overlap_topk = torch.sum(base_intopk * ptb_intopk, dim=-1)

    return torch.mean(overlap_topk, dim=-1) / k

def activation_similarity(outputs_base, outputs_perturb):
    base_hidden = outputs_base.hidden_states
    ptb_hidden = outputs_perturb.hidden_states

    ret = {}

    for i, _ in enumerate(zip(base_hidden, ptb_hidden)):
        hidden_base_i = torch.flatten(base_hidden[i][:, :-1, :], start_dim = 1)
        hidden_ptb_i = torch.flatten(ptb_hidden[i][:, :-1, :], start_dim = 1)

        cos_sim = torch.cosine_similarity(hidden_base_i, hidden_ptb_i, dim=-1).clamp(-1, 1)
        ret[f'activation_cos_sim_layer_{i}'] = cos_sim

        l2 = torch.sum((hidden_base_i - hidden_ptb_i) ** 2, dim=-1)
        ret[f'activation_l2_dist_layer_{i}'] = l2

    return ret