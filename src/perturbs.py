"""
File for all perturbation implementations.
Outside code should only ever call function `perturb`
"""

import re
import string

try:
    import nltk
    from nltk.corpus import wordnet as wn

    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

# QWERTY keyboard adjacency map for realistic typo simulation
QWERTY_ADJACENT = {
    "q": ["w", "a"],
    "w": ["q", "e", "s", "a"],
    "e": ["w", "r", "d", "s"],
    "r": ["e", "t", "f", "d"],
    "t": ["r", "y", "g", "f"],
    "y": ["t", "u", "h", "g"],
    "u": ["y", "i", "j", "h"],
    "i": ["u", "o", "k", "j"],
    "o": ["i", "p", "l", "k"],
    "p": ["o", "[", "l"],
    "a": ["q", "w", "s", "z"],
    "s": ["w", "e", "a", "d", "x", "z"],
    "d": ["e", "r", "s", "f", "c", "x"],
    "f": ["r", "t", "d", "g", "v", "c"],
    "g": ["t", "y", "f", "h", "b", "v"],
    "h": ["y", "u", "g", "j", "n", "b"],
    "j": ["u", "i", "h", "k", "m", "n"],
    "k": ["i", "o", "j", "l", "m"],
    "l": ["o", "p", "k", ";"],
    "z": ["a", "s", "x"],
    "x": ["s", "d", "z", "c"],
    "c": ["d", "f", "x", "v"],
    "v": ["f", "g", "c", "b"],
    "b": ["g", "h", "v", "n"],
    "n": ["h", "j", "b", "m"],
    "m": ["j", "k", "n"],
}

# Common punctuation typos (adjacent on keyboard or commonly confused)
PUNCT_TYPOS = {
    ".": [",", "?", "!"],
    ",": [".", ";"],
    "?": [".", "!"],
    "!": [".", "?"],
    ";": [",", ":"],
    ":": [";", "."],
    "'": ['"', "`"],
    '"': ["'", "`"],
}


def perturb(texts, perturb_pct, rng, ptb_type, tokenizer, model=None):
    if ptb_type == "char":
        return character_substitution(texts, perturb_pct, rng)
    elif ptb_type == "token":
        return token_substitution(texts, perturb_pct, rng, tokenizer)
    elif ptb_type == "word":
        return word_substitution(texts, perturb_pct, rng, tokenizer)
    elif ptb_type == "shuffle":
        return token_shuffle(texts, perturb_pct, rng, tokenizer=tokenizer)
    elif ptb_type == "typo":
        return typo_perturbation(texts, perturb_pct, rng)
    elif ptb_type == "synonym":
        return synonym_substitution(texts, perturb_pct, rng, tokenizer)
    elif ptb_type == "adv":
        return adversarial_token_substitution(
            texts, perturb_pct, rng, tokenizer, model=model
        )
    else:
        raise TypeError(
            "ptb_type must be one of ['char', 'token', 'word', 'shuffle', 'typo', 'synonym', 'adv']"
        )


def character_substitution(texts, perturb_pct, rng):
    """
    Substitute random ASCII characters in the text input.
    """
    sub_pool = string.ascii_letters + string.digits + string.punctuation
    ret = []
    for text in texts:
        word = []
        for c in text:
            if rng.randint(1, 100) < perturb_pct:
                word.append(rng.choice(sub_pool))
            else:
                word.append(c)
        ret.append("".join(word))
    return ret


def typo_perturbation(texts, perturb_pct, rng):
    """
    Introduce realistic typos based on QWERTY keyboard adjacency.
    """
    ret = []
    for text in texts:
        chars = list(text)
        for i, c in enumerate(chars):
            if c.lower() in QWERTY_ADJACENT and rng.randint(1, 100) <= perturb_pct:
                adj = QWERTY_ADJACENT[c.lower()]
                replacement = rng.choice(adj)
                # Preserve capitalization
                if c.isupper():
                    replacement = replacement.upper()
                chars[i] = replacement
            elif c in PUNCT_TYPOS and rng.randint(1, 100) <= perturb_pct:
                chars[i] = rng.choice(PUNCT_TYPOS[c])
        ret.append("".join(chars))
    return ret


def token_substitution(texts, perturb_pct, rng, tokenizer, max_length=128):
    """
    Substitute random tokens from the tokenizer vocabulary.
    """
    encodings = tokenizer(
        texts, truncation=True, max_length=max_length, add_special_tokens=False
    )

    ret = []
    for input_ids in encodings["input_ids"]:
        new_ids = [
            (
                rng.randint(0, tokenizer.vocab_size - 1)
                if rng.randint(1, 100) < perturb_pct
                else tid
            )
            for tid in input_ids
        ]
        ret.append(tokenizer.decode(new_ids))

    return ret


# Cache for word vocabulary (built once, shared across calls)
_WORD_VOCAB_CACHE = {}


def _get_word_vocab(tokenizer):
    """Build and cache a word-level vocabulary from the tokenizer."""
    cache_key = (
        tokenizer.name_or_path
        if hasattr(tokenizer, "name_or_path")
        else str(id(tokenizer))
    )

    if cache_key in _WORD_VOCAB_CACHE:
        return _WORD_VOCAB_CACHE[cache_key]

    # Build vocabulary by decoding first 10000 tokens and extracting alphabetic words
    word_set = set()
    for i in range(min(10000, tokenizer.vocab_size)):
        decoded = tokenizer.decode([i]).strip()
        if len(decoded) > 1 and decoded.isalpha():
            word_set.add(decoded)

    # Fallback to scanning full vocab if not enough words found
    if len(word_set) < 1000:
        for token, tid in tokenizer.vocab.items():
            decoded = tokenizer.decode([tid]).strip()
            if len(decoded) > 1 and decoded.isalpha():
                word_set.add(decoded)

    word_vocab = list(word_set)
    _WORD_VOCAB_CACHE[cache_key] = word_vocab
    return word_vocab


def word_substitution(texts, perturb_pct, rng, tokenizer, max_length=128):
    """
    Substitute random WORDS (whitespace-separated) from the tokenizer vocabulary.
    """
    word_vocab = _get_word_vocab(tokenizer)

    ret = []
    for text in texts:
        words = text.split()
        n_words = len(words)
        n_to_replace = max(1, int(perturb_pct * n_words / 100))

        # Select random word positions to replace
        positions = rng.sample(range(n_words), min(n_to_replace, n_words))

        words_list = list(words)
        for pos in positions:
            words_list[pos] = rng.choice(word_vocab)

        ret.append(" ".join(words_list))

    return ret


# Cache of word -> WordNet synonyms (built lazily, shared across calls)
_SYNONYM_CACHE = {}
_SYNONYM_INIT = False


def _ensure_wordnet():
    """Download the WordNet corpus once if nltk is available."""
    global _SYNONYM_INIT
    if _SYNONYM_INIT:
        return
    if _HAS_NLTK:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
    _SYNONYM_INIT = True


def _get_synonyms(word):
    """Return a sorted list of WordNet synonyms for a lowercase word (may be empty)."""
    if word in _SYNONYM_CACHE:
        return _SYNONYM_CACHE[word]
    syns = set()
    if _HAS_NLTK:
        for ss in wn.synsets(word):
            for lemma in ss.lemmas():
                name = lemma.name().replace("_", " ")
                # Keep single-token replacements only so token counts are preserved.
                if " " not in name and name.lower() != word:
                    syns.add(name)
    result = sorted(syns)
    _SYNONYM_CACHE[word] = result
    return result


def synonym_substitution(texts, perturb_pct, rng, tokenizer=None, max_length=128):
    """
    Replace words with WordNet synonyms (same synset members) at random positions.
    A more realistic word-level baseline than uniform random word replacement.
    Falls back to leaving a word untouched when no synonym exists.
    """
    if not _HAS_NLTK:
        raise ImportError(
            "synonym_substitution requires nltk + WordNet: "
            "pip install nltk && python -c \"import nltk; nltk.download('wordnet')\""
        )
    _ensure_wordnet()

    word_pattern = re.compile(r"^(\W*)(\w+)(\W*)$")

    def replace_with_synonym(word):
        m = word_pattern.match(word)
        if not m:
            return word
        pre, core, post = m.groups()
        syns = _get_synonyms(core.lower())
        if not syns:
            return word
        repl = rng.choice(syns)
        if core.isupper():
            repl = repl.upper()
        elif core[:1].isupper():
            repl = repl[:1].upper() + repl[1:]
        return pre + repl + post

    ret = []
    for text in texts:
        words = text.split()
        n_words = len(words)
        n_to_replace = max(1, int(perturb_pct * n_words / 100))
        positions = rng.sample(range(n_words), min(n_to_replace, n_words))

        words_list = list(words)
        for pos in positions:
            words_list[pos] = replace_with_synonym(words_list[pos])
        ret.append(" ".join(words_list))
    return ret


def token_shuffle(texts, perturb_pct, rng, tokenizer=None, max_length=128):
    """
    Scramble tokens in the BPE-tokenized sequence.
    """
    if tokenizer is None:
        # Fallback: whitespace-based (original behavior)
        ret = []
        for text in texts:
            toks = text.split()
            shuffle_window = int(perturb_pct * len(toks) / 100)
            start = (
                rng.randint(0, len(toks) - shuffle_window - 1)
                if perturb_pct < 100
                else 0
            )
            tok_to_shuffle = toks[start : start + shuffle_window]
            rng.shuffle(tok_to_shuffle)
            new_toks = toks[:start] + tok_to_shuffle + toks[start + shuffle_window :]
            ret.append(" ".join(new_toks))
        return ret

    # BPE token-level shuffling
    encodings = tokenizer(
        texts, truncation=True, max_length=max_length, add_special_tokens=False
    )

    ret = []
    for input_ids in encodings["input_ids"]:
        n_tokens = len(input_ids)
        shuffle_window = max(1, int(perturb_pct * n_tokens / 100))

        # Select contiguous window
        if perturb_pct < 100:
            start = rng.randint(0, n_tokens - shuffle_window)
        else:
            start = 0
            shuffle_window = n_tokens

        window = list(input_ids[start : start + shuffle_window])
        rng.shuffle(window)

        new_ids = (
            list(input_ids[:start]) + window + list(input_ids[start + shuffle_window :])
        )
        ret.append(tokenizer.decode(new_ids))
    return ret


# Cache of per-text token frequency ranks (adversarial targeting).
# Keyed by tuple of texts so the corpus frequency table is built once.
_ADV_FREQ_CACHE = {}


def _adv_targets(texts, rng, tokenizer, max_length=128):
    """Per-text, per-token adversarial saliency = corpus token frequency.

    Returns a list of (input_ids, numpy array of per-token frequency ranks).
    Frequencies are counted over the whole corpus once and cached, so the
    adversarial perturbation needs no model forward pass at all.
    """
    import numpy as np
    from collections import Counter

    key = tuple(texts)
    if key in _ADV_FREQ_CACHE:
        return _ADV_FREQ_CACHE[key]

    encodings = tokenizer(
        texts, truncation=True, max_length=max_length, add_special_tokens=False
    )
    all_ids = encodings["input_ids"]
    freq = Counter(tid for ids in all_ids for tid in ids)

    ret = [
        (ids, np.array([freq[tid] for tid in ids], dtype=np.float32)) for ids in all_ids
    ]
    _ADV_FREQ_CACHE[key] = ret
    return ret


def adversarial_token_substitution(
    texts, perturb_pct, rng, tokenizer, model=None, max_length=128
):
    """
    Adversarial token substitution (RQ4): targets the tokens that appear most
    frequently in the corpus (function words like 'the' carry the most
    probability mass) and replaces exactly that many tokens with random
    vocabulary tokens.

    This is the adversarial arm of the RQ4 experiment. Its control is the
    existing 'token' perturbation: same percentage (same expected number of
    replaced tokens, i.e. frequency-matched) but with positions chosen
    uniformly at random. No model forward pass is required.
    """
    targets = _adv_targets(texts, rng, tokenizer, max_length=max_length)

    ret = []
    for ids, freq in targets:
        n_tokens = len(ids)
        n_to_replace = max(1, int(perturb_pct * n_tokens / 100))
        n_to_replace = min(n_to_replace, n_tokens)
        # Adversarial: replace the positions whose tokens are most frequent in
        # the corpus. Ties are broken randomly (deterministically via rng).
        positions = sorted(range(n_tokens), key=lambda j: (-freq[j], rng.random()))[
            :n_to_replace
        ]
        pos_set = set(positions)
        new_ids = [
            rng.randint(0, tokenizer.vocab_size - 1) if j in pos_set else tid
            for j, tid in enumerate(ids)
        ]
        ret.append(tokenizer.decode(new_ids))
    return ret
