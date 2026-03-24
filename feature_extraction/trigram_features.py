"""
trigram_features.py

Computes trigram surprisal features for (reference, variant) pairs.

Reference:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

Paper §Surprisal:
    "we used a traditional trigram language model to estimate the
     surprisal of the target word conditioned only on the two
     preceding context words. Sentence-level surprisal is computed
     by summing per-word surprisal scores."

Model:
    NLTK MLE trigram model (class nltk.lm.models.MLE).

    model.score(w, [w1, w2])  → P(w | w1, w2)  in range [0, 1]

Smoothing / fallback strategy:
    MLE returns exactly 0.0 for unseen ngrams (no smoothing).
    We use a three-level backoff:
        1. Trigram  P(w3 | w1, w2)
        2. Bigram   P(w3 | w2)        if trigram is 0
        3. Unigram  P(w3)             if bigram is also 0
        4. Epsilon  1e-12             if all three are 0 (truly OOV)

Delta direction (paper convention):
    delta_trigram = trigram_reference − trigram_variant
    Negative delta → reference has LOWER surprisal (more predictable)
    → model correctly prefers the reference sentence.
"""

import math
import pickle


# ─────────────────────────────────────────────────────────────
# load_trigram_model
# ─────────────────────────────────────────────────────────────
def load_trigram_model(path):
    """Load a pickled NLTK MLE trigram model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# _trigram_prob
#
# Returns P(w3 | w1, w2) with three-level backoff.
# Never returns 0 — minimum is epsilon (1e-12).
# ─────────────────────────────────────────────────────────────
def _trigram_prob(model, w1, w2, w3):
    """
    Fetch trigram probability with backoff:
        1. Trigram  P(w3 | w1, w2)
        2. Bigram   P(w3 | w2)
        3. Unigram  P(w3)
        4. Epsilon  1e-12

    model.score() returns a float in [0, 1].
    0.0 means the ngram was never seen in training (MLE, no smoothing).
    """
    # Level 1: trigram
    prob = model.score(w3, [w1, w2])
    if prob > 0:
        return prob

    # Level 2: bigram backoff
    prob = model.score(w3, [w2])
    if prob > 0:
        return prob

    # Level 3: unigram backoff
    prob = model.score(w3)
    if prob > 0:
        return prob

    # Level 4: epsilon for completely OOV words
    return 1e-12


# ─────────────────────────────────────────────────────────────
# per_word_trigram_surprisal
#
# Returns a list of (word, probability, surprisal_nats) tuples
# for every word in the sentence that has trigram context.
# Words at positions 0 and 1 are returned with prob=None and
# surprisal=None (no full trigram context available yet).
#
# Used by:
#   - sentence_trigram_surprisal() to sum sentence-level score
#   - test_trigram.py to inspect word-level scores
# ─────────────────────────────────────────────────────────────
def per_word_trigram_surprisal(sentence_str, model):
    """
    Compute per-word trigram surprisal for a sentence string.

    Returns list of dicts:
        {
            "position":   int,    # 1-based word position
            "word":       str,    # surface form
            "context":    str,    # "w1 w2" context used
            "prob":       float,  # P(word | context) — None if no context
            "backoff":    str,    # "trigram" / "bigram" / "unigram" / "epsilon"
            "surprisal":  float,  # -ln(prob) in nats — None if no context
        }

    Words at positions 1 and 2 have no full trigram context.
    They are included in the output with prob=None, surprisal=None.
    """
    words  = sentence_str.split()
    result = []

    for i, word in enumerate(words):
        if i < 2:
            # No trigram context available for first two words
            result.append({
                "position":  i + 1,
                "word":      word,
                "context":   "",
                "prob":      None,
                "backoff":   "no context",
                "surprisal": None,
            })
            continue

        w1, w2, w3 = words[i-2], words[i-1], word

        # Determine which backoff level was used
        prob = model.score(w3, [w1, w2])
        if prob > 0:
            backoff = "trigram"
        else:
            prob = model.score(w3, [w2])
            if prob > 0:
                backoff = "bigram"
            else:
                prob = model.score(w3)
                if prob > 0:
                    backoff = "unigram"
                else:
                    prob    = 1e-12
                    backoff = "epsilon (OOV)"

        surprisal = -math.log(prob)

        result.append({
            "position":  i + 1,
            "word":      word,
            "context":   f"{w1} {w2}",
            "prob":      prob,
            "backoff":   backoff,
            "surprisal": surprisal,
        })

    return result


# ─────────────────────────────────────────────────────────────
# sentence_trigram_surprisal
#
# Sentence-level surprisal = sum of per-word surprisals.
# Paper: "summing per-word surprisal scores within each sentence"
# ─────────────────────────────────────────────────────────────
def sentence_trigram_surprisal(sentence_str, model):
    """
    Compute total trigram surprisal for a sentence string.

    surprisal(w_i) = -log P(w_i | w_{i-2}, w_{i-1})   [nats]
    total = sum over i from 2 to len(words)-1

    Words at positions 0 and 1 contribute 0 (no full trigram context).
    Returns a non-negative float.
    """
    words = sentence_str.split()

    if len(words) < 3:
        return 0.0

    total = 0.0
    for i in range(2, len(words)):
        prob       = _trigram_prob(model, words[i-2], words[i-1], words[i])
        total     += -math.log(prob)

    return total


# ─────────────────────────────────────────────────────────────
# compute_trigram_features
#
# Adds trigram surprisal features to every dataset item.
#
# Adds:
#   "trigram_reference" — sentence-level surprisal of reference
#   "trigram_variant"   — sentence-level surprisal of variant
#   "delta_trigram"     — trigram_reference − trigram_variant
#
# Paper convention: delta = ref − var.
# Negative delta = reference is more predictable = preferred.
# ─────────────────────────────────────────────────────────────
def compute_trigram_features(dataset, model):
    """Compute trigram surprisal delta for all (reference, variant) pairs."""

    results = []

    for item in dataset:

        s_ref = sentence_trigram_surprisal(item["reference"], model)
        s_var = sentence_trigram_surprisal(item["variant"],   model)

        # Paper: delta = reference − variant
        delta = s_ref - s_var

        results.append({
            **item,
            "trigram_reference": s_ref,
            "trigram_variant":   s_var,
            "delta_trigram":     delta,
        })

    return results