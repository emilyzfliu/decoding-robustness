"""
Shared gradient-guided (HotFlip-style) token attack.

Used by both the context-insertion `adversarial` condition (attacks a spliced-in
placeholder span) and the question-level `adversarial_swap` condition (attacks
existing context tokens in place). Both just supply different `positions`.

Reference: Ebrahimi et al. 2018 "HotFlip"; Wallace et al. 2019 "Universal
Adversarial Triggers" (same embedding-gradient first-order candidate scoring,
verified with real forward passes rather than trusted blindly).
"""
import torch


def _sequence_loss(logits, input_ids, attention_mask):
    """Per-sequence average CE loss, masked like src/eval.py:nll()."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = torch.nn.CrossEntropyLoss(reduction='none')(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(input_ids.size(0), -1)
    mask = (shift_labels != -100).float()
    return (token_losses * mask).sum(dim=1) / mask.sum(dim=1)


def hotflip_attack(model, tokenizer, input_ids, attention_mask, positions, device,
                    rng, n_candidates=50, n_iters=20, mode="maximize_loss"):
    """
    Greedy, gradient-shortlisted token attack on a single sequence.

    input_ids, attention_mask: 1D tensors (single sequence, no batch dim).
    positions: list[int] of token indices in input_ids that may be modified.
    rng: random.Random instance, used to pick which position to attack each
        iteration (round-robin through a shuffled order of `positions`).
    mode: only "maximize_loss" is implemented (increase the model's own NLL
        on this sequence).

    Each iteration:
      1. Embed the sequence (requires_grad), forward pass, compute masked
         sequence loss, backward.
      2. Take the gradient at the single attacked position, dot it against
         the full embedding matrix to shortlist n_candidates tokens via the
         first-order Taylor approximation of how much each substitution
         would change the loss.
      3. Build a batch of candidate sequences (one per shortlisted token),
         forward pass (no grad) to get their *actual* loss.
      4. Commit the best candidate only if it improves on the current loss;
         otherwise leave that position unchanged this round.

    Returns the (possibly modified) input_ids tensor (1D, same shape as input).
    """
    if mode != "maximize_loss":
        raise ValueError("Only mode='maximize_loss' is implemented")
    if not positions:
        return input_ids

    embedding_matrix = model.get_input_embeddings().weight  # (V, d)
    vocab_size = embedding_matrix.size(0)

    input_ids = input_ids.clone().to(device)
    attention_mask = attention_mask.to(device)

    attack_order = list(positions)
    rng.shuffle(attack_order)

    was_training = model.training
    model.eval()

    for step in range(n_iters):
        pos = attack_order[step % len(attack_order)]

        # 1. Gradient of the current sequence's loss w.r.t. its embeddings.
        embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach().clone()
        embeds.requires_grad_(True)
        outputs = model(inputs_embeds=embeds, attention_mask=attention_mask.unsqueeze(0))
        current_loss = _sequence_loss(outputs.logits, input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        model.zero_grad(set_to_none=True)
        current_loss.sum().backward()
        grad_at_pos = embeds.grad[0, pos]  # (d,)

        # 2. Shortlist candidates by first-order approximation: maximize grad . e_cand.
        with torch.no_grad():
            scores = embedding_matrix @ grad_at_pos  # (V,)
            current_token = input_ids[pos].item()
            scores[current_token] = -float("inf")
            if tokenizer.pad_token_id is not None:
                scores[tokenizer.pad_token_id] = -float("inf")
            top_candidates = torch.topk(scores, k=min(n_candidates, vocab_size)).indices

            # 3. Verify candidates with real forward passes (batched).
            n_cand = top_candidates.size(0)
            batch_ids = input_ids.unsqueeze(0).repeat(n_cand, 1)
            batch_ids[:, pos] = top_candidates
            batch_mask = attention_mask.unsqueeze(0).repeat(n_cand, 1)

            batch_outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            candidate_losses = _sequence_loss(batch_outputs.logits, batch_ids, batch_mask)

            best_idx = torch.argmax(candidate_losses)
            best_loss = candidate_losses[best_idx].item()

            # 4. Commit only if it's actually an improvement.
            if best_loss > current_loss.item():
                input_ids = batch_ids[best_idx].clone()

    if was_training:
        model.train()

    return input_ids.detach().cpu()
