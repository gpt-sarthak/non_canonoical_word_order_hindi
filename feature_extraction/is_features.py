"""
is_features.py

Information Status (IS) / Givenness feature extraction.

Reference:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

Paper definition (Section: Information Status):
-----------------------------------------------
"We annotate each sentence in our dataset with Information Status (IS).
 For this, we examine subject and object phrases in the target sentences,
 checking if any content words within these phrases are mentioned in the
 preceding context sentence. If so, these phrases are tagged as GIVEN,
 otherwise they are tagged as NEW. Additionally, we label phrases as GIVEN
 if the head of these constituents is a pronoun."

Numerical scores:
    Given-New  order  → +1
    New-Given  order  → -1
    Given-Given order → 0
    New-New    order  → 0

IS score is computed separately for the reference and variant sentences
because they place subject and object in different positions.

delta_is = is_reference − is_variant
"""

from data.hutb_loader import (
    CONTENT_POS,
    SUBJECT_RELS,
    OBJECT_RELS,
    subtree_tokens,
)


# ------------------------------------------------------------
# _content_words_of_phrase
#
# Returns the set of lowercased lemmas of content words within
# the dependency subtree rooted at `head_token`.
#
# Paper: "content words within these phrases"
# Content words = NOUN, PROPN, VERB, ADJ, ADV, NUM
# (function words such as postpositions and particles excluded)
# ------------------------------------------------------------
def _content_words_of_phrase(sentence, head_token):

    phrase_tokens = subtree_tokens(sentence, head_token["id"])

    return {
        t["lemma"].lower()
        for t in phrase_tokens
        if t["upos"] in CONTENT_POS
    }


# ------------------------------------------------------------
# _is_given
#
# A phrase is GIVEN if:
#   (a) its head token is a pronoun (UPOS == PRON), OR
#   (b) at least one content word in its subtree appears in the
#       context sentence word set.
#
# Paper: "Additionally, we label phrases as GIVEN if the head
#         of these constituents is a pronoun."
# ------------------------------------------------------------
def _is_given(sentence, head_token, context_words):

    # (a) Pronoun head
    if head_token["upos"] == "PRON":
        return True

    # (b) Content word overlap with context
    phrase_content = _content_words_of_phrase(sentence, head_token)
    return bool(phrase_content & context_words)


# ------------------------------------------------------------
# _context_word_set
#
# Returns the set of lowercased words from the context sentence
# string (all words, not just content words, to be maximally
# inclusive — matching the paper's phrasing "mentioned in the
# preceding context sentence").
# ------------------------------------------------------------
def _context_word_set(context_str):
    if not context_str or not context_str.strip():
        return set()
    return {w.lower() for w in context_str.split()}


# ------------------------------------------------------------
# _is_score_for_sentence
#
# Computes the IS score for one sentence (reference or variant),
# given the dependency parse of the REFERENCE sentence and the
# word order applied to it.
#
# Arguments:
#   sentence      — list of token dicts (original treebank parse)
#   word_order    — list of token_ids in the sentence's surface order
#                   (for the reference this is [t["id"] for t in sentence];
#                    for a variant it is the permuted order)
#   context_words — set of lowercased words from the context sentence
#
# Returns:
#   +1  if the phrase that appears FIRST (leftmost) is GIVEN and
#       the phrase that appears SECOND (rightmost) is NEW
#   -1  if the phrase that appears FIRST is NEW and SECOND is GIVEN
#    0  if both phrases have the same givenness (Given-Given or New-New)
# ------------------------------------------------------------
def _is_score_for_sentence(sentence, word_order, context_words):

    root = next((t for t in sentence if t["head"] == 0), None)
    if root is None:
        return 0

    root_id = root["id"]

    # Find the subject and object head tokens
    subj_token = next(
        (t for t in sentence if t["deprel"] in SUBJECT_RELS and t["head"] == root_id),
        None,
    )
    obj_token = next(
        (t for t in sentence if t["deprel"] in OBJECT_RELS and t["head"] == root_id),
        None,
    )

    if subj_token is None or obj_token is None:
        return 0

    subj_given = _is_given(sentence, subj_token, context_words)
    obj_given  = _is_given(sentence, obj_token,  context_words)

    # Determine which phrase appears first in the surface order
    # word_order maps position → token_id; invert to token_id → position
    position = {tok_id: pos for pos, tok_id in enumerate(word_order)}

    subj_pos = position.get(subj_token["id"], 0)
    obj_pos  = position.get(obj_token["id"],  0)

    if subj_pos < obj_pos:
        first_given, second_given = subj_given, obj_given
    else:
        first_given, second_given = obj_given, subj_given

    # Assign numerical score
    if first_given and not second_given:
        return +1   # Given-New
    if not first_given and second_given:
        return -1   # New-Given
    return 0        # Given-Given or New-New


# ------------------------------------------------------------
# compute_is_features
#
# Adds IS features to every item in the dataset.
#
# For each (reference, variant) pair:
#   is_reference — IS score of the reference word order
#   is_variant   — IS score of the variant word order
#   delta_is     — is_reference − is_variant
#
# Note on delta direction:
#   Paper footnote 6: "we expect the reference-variant difference
#   [(+1) - (-1) = 2] for IS score to be a positive value if
#   givenness is truly adhered to."
#   Therefore delta_is = IS_ref − IS_var  (not var − ref).
# ------------------------------------------------------------
def compute_is_features(dataset):

    results = []

    for item in dataset:

        sentence   = item["tokens"]        # original parse
        reference  = item["reference"]
        context    = item.get("context", "")

        context_words = _context_word_set(context)

        # Reference word order (original treebank order)
        ref_order = [t["id"] for t in sentence]

        # Variant word order (permuted)
        var_order = item["order"]

        is_ref = _is_score_for_sentence(sentence, ref_order, context_words)
        is_var = _is_score_for_sentence(sentence, var_order, context_words)

        # delta = reference − variant
        # (positive when reference follows given-before-new more than variant)
        delta_is = is_ref - is_var

        results.append({
            **item,
            "is_reference": is_ref,
            "is_variant":   is_var,
            "delta_is":     delta_is,
        })

    return results