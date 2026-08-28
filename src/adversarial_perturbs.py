"""
Adversarial / injection perturbations: context insertion and question-level
attacks. Separate from src/perturbs.py because the 'adversarial' conditions
need the model itself (for gradient-guided attacks via src/hotflip.py), unlike
the existing char/token/word/shuffle/typo perturbations which only need text
+ tokenizer.

Each adversarial condition ('adversarial', 'adversarial_swap') has a matched
uniform-random control ('random_insertion', 'random_swap' respectively) that
edits the same span/positions and edit budget but with randomly-sampled
replacement tokens instead of gradient-guided ones — isolating whether
adversarial optimization matters beyond just introducing out-of-distribution
tokens at that location.

Public entry points: `context_insertion`, `question_perturbation`.
"""
import re                  

import torch

from src.hotflip import hotflip_attack


# ---------------------------------------------------------------------------
# Context insertion
# ---------------------------------------------------------------------------

def context_insertion(texts, condition, rng, tokenizer, model=None, device=None,
                       distractor_pool=None, insertion_len=8, n_iters=20, max_length=1024):
    if condition == 'clean':
        return list(texts)
    elif condition == 'topic_shift':
        return _topic_shift(texts, rng, distractor_pool)
    elif condition == 'misleading_claim':
        return _misleading_claim(texts, rng)
    elif condition == 'random_insertion':
        return _random_insertion(texts, rng, tokenizer, insertion_len, max_length)
    elif condition == 'adversarial':
        return _adversarial_insertion(texts, rng, tokenizer, model, device, insertion_len, n_iters, max_length)
    else:
        raise ValueError(
            "condition must be one of ['clean', 'topic_shift', 'misleading_claim', "
            "'random_insertion', 'adversarial']"
        )


def _split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _topic_shift(texts, rng, distractor_pool):
    """Splice an unrelated sentence, sampled from a different passage, into the midpoint."""
    if not distractor_pool:
        raise ValueError("distractor_pool (a list of other passages) is required for 'topic_shift'")
    ret = []
    for text in texts:
        pool = [t for t in distractor_pool if t != text] or distractor_pool
        distractor_text = rng.choice(pool)
        sentences = _split_sentences(distractor_text)
        distractor_sentence = rng.choice(sentences) if sentences else distractor_text.strip()[:200]

        words = text.split()
        if len(words) < 4:
            ret.append(text + ' ' + distractor_sentence)
            continue
        mid = len(words) // 2
        new_words = words[:mid] + [distractor_sentence] + words[mid:]
        ret.append(' '.join(new_words))
    return ret


_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')
_NUMBER_RE = re.compile(r'\b\d+\b')
_CAPITALIZED_RE = re.compile(r'\b[A-Z][a-z]+\b')
_COPULA_RE = re.compile(r'\b(is|was|were|has|have|are)\b')


def _misleading_claim(texts, rng):
    """
    Corrupt the passage itself to create an internally-plausible but false claim:
    1) mutate a year/number if one exists, else
    2) swap two capitalized (naive proper-noun) tokens, else
    3) negate the first copula/aux verb as a last resort.
    """
    ret = []
    for text in texts:
        new_text = _corrupt_numbers(text, rng)
        if new_text is None:
            new_text = _corrupt_proper_nouns(text, rng)
        if new_text is None:
            new_text = _negate_copula(text)
        ret.append(new_text if new_text is not None else text)
    return ret


def _corrupt_numbers(text, rng):
    year_matches = list(_YEAR_RE.finditer(text))
    if year_matches:
        m = rng.choice(year_matches)
        old_val = int(m.group())
        delta = rng.choice([-1, 1]) * rng.randint(10, 200)
        new_val = max(1, old_val + delta)
        return text[:m.start()] + str(new_val) + text[m.end():]

    num_matches = [m for m in _NUMBER_RE.finditer(text) if not _YEAR_RE.fullmatch(m.group())]
    if num_matches:
        m = rng.choice(num_matches)
        old_val = int(m.group())
        delta = rng.randint(1, max(1, old_val)) * rng.choice([-1, 1])
        new_val = max(0, old_val + delta)
        if new_val == old_val:
            new_val += 1
        return text[:m.start()] + str(new_val) + text[m.end():]

    return None


def _corrupt_proper_nouns(text, rng):
    distinct = list({m.group() for m in _CAPITALIZED_RE.finditer(text)})
    if len(distinct) < 2:
        return None
    a, b = rng.sample(distinct, 2)
    placeholder = "SWAP"
    new_text = re.sub(rf'\b{re.escape(a)}\b', placeholder, text, count=1)
    new_text = re.sub(rf'\b{re.escape(b)}\b', a, new_text, count=1)
    return new_text.replace(placeholder, b)


def _negate_copula(text):
    m = _COPULA_RE.search(text)
    if not m:
        return None
    return text[:m.end()] + ' not' + text[m.end():]


def _adversarial_insertion(texts, rng, tokenizer, model, device, insertion_len, n_iters, max_length):
    """Splice a placeholder span into the passage, then HotFlip-optimize it to maximize NLL."""
    if model is None:
        raise ValueError("model is required for the 'adversarial' condition")

    placeholder_token_id = tokenizer.encode(' the', add_special_tokens=False)[0]
    ret = []
    for text in texts:
        base_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        keep_budget = max(1, max_length - insertion_len)
        base_ids = base_ids[:keep_budget]

        mid = len(base_ids) // 2
        insertion_ids = [placeholder_token_id] * insertion_len
        combined = (base_ids[:mid] + insertion_ids + base_ids[mid:])[:max_length]
        inserted_positions = [p for p in range(mid, mid + insertion_len) if p < len(combined)]

        input_ids = torch.tensor(combined, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        attacked_ids = hotflip_attack(
            model, tokenizer, input_ids, attention_mask, inserted_positions,
            device, rng, n_candidates=50, n_iters=n_iters,
        )
        ret.append(tokenizer.decode(attacked_ids))
    return ret


def _sample_random_tokens(rng, vocab_size, n, exclude):
    out = []
    for _ in range(n):
        tid = rng.randrange(vocab_size)
        while tid in exclude:
            tid = rng.randrange(vocab_size)
        out.append(tid)
    return out


def _random_insertion(texts, rng, tokenizer, insertion_len, max_length):
    """Uniform-random control for 'adversarial': splices a span of the same
    length at the same midpoint position, but the inserted tokens are sampled
    uniformly at random instead of HotFlip-optimized. Isolates whether
    gradient-guided token choice matters, independent of splice location/length."""
    exclude = {tid for tid in (tokenizer.pad_token_id, tokenizer.eos_token_id) if tid is not None}
    ret = []
    for text in texts:
        base_ids = tokenizer(text, add_special_tokens=False)['input_ids']
        keep_budget = max(1, max_length - insertion_len)
        base_ids = base_ids[:keep_budget]

        mid = len(base_ids) // 2
        insertion_ids = _sample_random_tokens(rng, tokenizer.vocab_size, insertion_len, exclude)
        combined = (base_ids[:mid] + insertion_ids + base_ids[mid:])[:max_length]
        ret.append(tokenizer.decode(combined))
    return ret


# ---------------------------------------------------------------------------
# Question-level perturbations
# ---------------------------------------------------------------------------

def question_perturbation(texts, condition, rng, tokenizer, model=None, device=None, max_length=1024):
    if condition == 'clean':
        return list(texts)
    elif condition == 'synonym':
        return _synonym_substitution(texts, rng)
    elif condition == 'reorder':
        return _clause_reorder(texts, rng)
    elif condition == 'negation_paraphrase':
        return _negation_paraphrase(texts, rng)
    elif condition == 'random_swap':
        return _random_swap(texts, rng, tokenizer, max_length)
    elif condition == 'adversarial_swap':
        return _adversarial_swap(texts, rng, tokenizer, model, device, max_length)
    else:
        raise ValueError(
            "condition must be one of ['clean', 'synonym', 'reorder', 'negation_paraphrase', "
            "'random_swap', 'adversarial_swap']"
        )


_WORDNET_READY = False


def _ensure_wordnet():
    global _WORDNET_READY
    if _WORDNET_READY:
        return
    import nltk
    for resource in ('corpora/wordnet', 'corpora/omw-1.4'):
        try:
            nltk.data.find(resource)
        except LookupError:
            name = resource.split('/')[-1]
            raise RuntimeError(
                f"NLTK resource '{name}' not found. Run "
                f"`python -m nltk.downloader wordnet omw-1.4` once before using the "
                f"'synonym' condition."
            )
    _WORDNET_READY = True


_ALPHA_RE = re.compile(r'^[A-Za-z]+$')


def _synonym_substitution(texts, rng, max_replace_frac=0.3):
    _ensure_wordnet()
    from nltk.corpus import wordnet as wn

    ret = []
    for text in texts:
        words = text.split()
        candidate_idx = [i for i, w in enumerate(words) if _ALPHA_RE.match(w)]
        rng.shuffle(candidate_idx)
        n_to_try = max(1, int(len(candidate_idx) * max_replace_frac))

        for idx in candidate_idx[:n_to_try]:
            word = words[idx]
            lemmas = set()
            for syn in wn.synsets(word.lower()):
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != word.lower():
                        lemmas.add(name)
            if not lemmas:
                continue
            replacement = rng.choice(list(lemmas))
            if word[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            words[idx] = replacement
        ret.append(' '.join(words))
    return ret


_CLAUSE_SPLIT_RE = re.compile(r'(,\s*(?:and|but|which|who|because|although|while)\s+|,\s+)', re.IGNORECASE)


def _clause_reorder(texts, rng):
    """Swap two adjacent comma/conjunction-delimited clauses (genuine syntactic change)."""
    ret = []
    for text in texts:
        parts = _CLAUSE_SPLIT_RE.split(text)
        clauses = parts[0::2]
        delimiters = parts[1::2]
        if len(clauses) < 2:
            ret.append(text)
            continue
        i = rng.randrange(len(clauses) - 1)
        clauses[i], clauses[i + 1] = clauses[i + 1], clauses[i]
        rebuilt = clauses[0]
        for d, c in zip(delimiters, clauses[1:]):
            rebuilt += d + c
        ret.append(rebuilt)
    return ret


_NEGATION_PAIRS = [
    (r'\bis not\b', "isn't"), (r"\bisn't\b", 'is not'),
    (r'\bdid not\b', "didn't"), (r"\bdidn't\b", 'did not'),
    (r'\bdoes not\b', "doesn't"), (r"\bdoesn't\b", 'does not'),
    (r'\bdo not\b', "don't"), (r"\bdon't\b", 'do not'),
    (r'\bcannot\b', 'can not'), (r'\bcan not\b', 'cannot'),
    (r'\bwill not\b', "won't"), (r"\bwon't\b", 'will not'),
    (r'\bwas not\b', "wasn't"), (r"\bwasn't\b", 'was not'),
    (r'\bwere not\b', "weren't"), (r"\bweren't\b", 'were not'),
    (r'\bnever\b', 'not ever'), (r'\bnot ever\b', 'never'),
]


def _negation_paraphrase(texts, rng):
    """Rule-based polarity-preserving rewrite (contraction <-> expanded form, etc.)."""
    ret = []
    for text in texts:
        candidates = list(_NEGATION_PAIRS)
        rng.shuffle(candidates)
        new_text = text
        applied = False
        for pattern, replacement in candidates:
            m = re.search(pattern, new_text, re.IGNORECASE)
            if m:
                new_text = new_text[:m.start()] + replacement + new_text[m.end():]
                applied = True
                break
        ret.append(new_text if applied else text)
    return ret


def _random_swap(texts, rng, tokenizer, max_length, n_edits=20):
    """Uniform-random control for 'adversarial_swap': the same number of context
    positions get modified (up to n_edits, matching HotFlip's n_iters budget and
    picked via the same shuffled-position order), but each replacement token is
    sampled uniformly at random instead of gradient-selected."""
    exclude = {tid for tid in (tokenizer.pad_token_id, tokenizer.eos_token_id) if tid is not None}
    ret = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']
        if len(ids) < 2:
            ret.append(text)
            continue

        attack_order = list(range(len(ids)))
        rng.shuffle(attack_order)
        for pos in attack_order[:min(n_edits, len(ids))]:
            [new_id] = _sample_random_tokens(rng, tokenizer.vocab_size, 1, exclude | {ids[pos]})
            ids[pos] = new_id
        ret.append(tokenizer.decode(ids))
    return ret


def _adversarial_swap(texts, rng, tokenizer, model, device, max_length):
    """HotFlip-attack the context tokens in place (the held-out final word isn't part of the input)."""
    if model is None:
        raise ValueError("model is required for the 'adversarial_swap' condition")

    ret = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']
        if len(ids) < 2:
            ret.append(text)
            continue
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        positions = list(range(len(ids)))

        attacked_ids = hotflip_attack(
            model, tokenizer, input_ids, attention_mask, positions,
            device, rng, n_candidates=50, n_iters=min(20, len(ids)),
        )
        ret.append(tokenizer.decode(attacked_ids))
    return ret
