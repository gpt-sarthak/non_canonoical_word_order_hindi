import os
import itertools

# Path to the Hindi Dependency Treebank (HDTB) training file
TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"


# ------------------------------------------------------------
# Function: load_conllu
# Purpose:
#   Reads the CoNLL-U formatted treebank file and converts it
#   into a list of sentences where each sentence is a list of
#   token dictionaries.
# ------------------------------------------------------------
def load_conllu(filepath):

    sentences = []
    current_sentence = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:

            line = line.strip()

            # Blank line indicates the end of a sentence
            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Skip comment lines beginning with '#'
            if line.startswith("#"):
                continue

            parts = line.split("\t")

            # Skip multiword tokens (e.g., 3-4)
            if "-" in parts[0]:
                continue

            # Skip empty nodes used in enhanced dependencies
            if "." in parts[0]:
                continue

            # Create a dictionary for each token
            token = {
                    "id": int(parts[0]),          # word position in sentence
                    "word": parts[1],             # surface form
                    "lemma": parts[2],            # lemma
                    "upos": parts[3],             # universal POS tag
                    "feats": parts[5],            # morphological features
                    "head": int(parts[6]),        # syntactic head index
                    "deprel": parts[7]            # dependency relation
            }

            current_sentence.append(token)

    return sentences


# ------------------------------------------------------------
# Function: is_projective
# Purpose:
#   Checks whether a dependency tree is projective.
#   A projective tree has no crossing dependency arcs.
#   Required by the paper's filtering criteria.
# ------------------------------------------------------------
def is_projective(sentence):

    arcs = []

    # Collect all dependency arcs
    for token in sentence:
        if token["head"] != 0:
            arcs.append((token["id"], token["head"]))

    # Check for crossing arcs
    for i1, h1 in arcs:
        for i2, h2 in arcs:

            a, b = sorted((i1, h1))
            c, d = sorted((i2, h2))

            if a < c < b < d or c < a < d < b:
                return False

    return True


# ------------------------------------------------------------
# Function: is_declarative
# Purpose:
#   Filters out interrogative sentences.
#   The paper restricts the dataset to declarative sentences.
# ------------------------------------------------------------
def is_declarative(sentence):
    for token in sentence:
        if token["word"] == "?":
            return False
    return True


# ------------------------------------------------------------
# Function: is_valid_sov_sentence
# Purpose:
#   Implements the filtering rules described in the paper:
#
#   1. Sentence must contain both subject and object
#   2. Dependency tree must be projective
#   3. Sentence must be declarative
#   4. Root must be a finite verb
#   5. Root must have at least two preverbal dependents
# ------------------------------------------------------------
def is_valid_treebank_sentence(sentence):

    root = None
    has_subject = False
    has_object = False

    for token in sentence:

        # Identify the root node
        if token["head"] == 0:
            root = token

        # Detect subject
        if token["deprel"] in ["nsubj", "csubj"]:
            has_subject = True

        # Detect object
        if token["deprel"] in ["obj", "iobj"]:
            has_object = True

    if root is None:
        return False

    # Root must be a verb, also checks if the root is a finite verb
    if root["upos"] != "VERB":
        return False
    
    # Tree must be projective
    if not is_projective(sentence):
        return False
    
    # Sentence must be declarative
    if not is_declarative(sentence):
        return False

    # Identify dependents that occur before the verb
    preverbal_dependents = [
    token for token in sentence
    if token["head"] == root["id"] and token["id"] < root["id"]
    ]

    # Require at least two preverbal dependents
    if len(preverbal_dependents) < 2:
        return False

    return has_subject and has_object


# ------------------------------------------------------------
# Function: get_subtree
# Purpose:
#   Recursively extracts the dependency subtree rooted at a
#   given token. Used to preserve syntactic phrases during
#   permutation.
# ------------------------------------------------------------
def get_subtree(sentence, token_id):

    subtree_ids = set()
    stack = [token_id]

    while stack:

        current = stack.pop()

        if current in subtree_ids:
            continue

        subtree_ids.add(current)

        # Add children of the current node
        for token in sentence:
            if token["head"] == current:
                stack.append(token["id"])

    subtree = []

    # Collect tokens belonging to the subtree
    for token in sentence:
        if token["id"] in subtree_ids:
            subtree.append(token)

    return subtree


# ------------------------------------------------------------
# Function: subtree_tokens
# Purpose:
#   Returns tokens of a subtree sorted by their position.
# ------------------------------------------------------------
def subtree_tokens(sentence, token_id):

    subtree = get_subtree(sentence, token_id)

    subtree = sorted(subtree, key=lambda x: x["id"])

    return subtree


# ------------------------------------------------------------
# Function: generate_variants_subtrees
# Purpose:
#   Generates alternative sentence orders by permuting the
#   preverbal phrases attached to the root verb.
#
#   Postverbal elements remain fixed.
#   Maximum number of variants is capped at 99 to avoid
#   combinatorial explosion.
# ------------------------------------------------------------
import itertools


import itertools

def generate_variants_subtrees(sentence, max_variants=99):

    root = None

    # ------------------------------------------------------------
    # Identify the root verb of the sentence
    # ------------------------------------------------------------
    for token in sentence:
        if token["head"] == 0:
            root = token
            break

    if root is None:
        return []

    root_id = root["id"]

    preverbal_phrases = []
    postverbal_phrases = []

    # ------------------------------------------------------------
    # Collect dependency subtrees of root dependents
    # ------------------------------------------------------------
    for token in sentence:

        if token["head"] == root_id:

            # Ignore punctuation
            if token["deprel"] == "punct":
                continue

            phrase_tokens = subtree_tokens(sentence, token["id"])

            # phrases before the verb are permuted
            if token["id"] < root_id:
                preverbal_phrases.append(phrase_tokens)

            # phrases after the verb remain fixed
            else:
                postverbal_phrases.append((token["id"], phrase_tokens))

    # ------------------------------------------------------------
    # Preserve order of postverbal phrases
    # ------------------------------------------------------------
    postverbal_phrases = sorted(postverbal_phrases, key=lambda x: x[0])

    postverbal_tokens = []
    for _, phrase in postverbal_phrases:
        postverbal_tokens.extend(phrase)

    variants = []

    # ------------------------------------------------------------
    # Generate permutations of preverbal phrases
    # ------------------------------------------------------------
    perms = itertools.permutations(preverbal_phrases)

    for perm in perms:

        new_tokens = []

        # Add permuted preverbal phrases
        for phrase in perm:
            new_tokens.extend(phrase)

        # Add the root verb
        new_tokens.append(root)

        # Add postverbal phrases
        new_tokens.extend(postverbal_tokens)

        # ------------------------------------------------------------
        # Ensure every token from the original sentence is present
        # (handles auxiliaries, punctuation, etc.)
        # ------------------------------------------------------------
        used_ids = {t["id"] for t in new_tokens}

        for token in sentence:
            if token["id"] not in used_ids:
                new_tokens.append(token)

        # IMPORTANT: DO NOT sort tokens here
        # Sorting would destroy the permutation order

        # ------------------------------------------------------------
        # Build sentence string
        # ------------------------------------------------------------
        words = [t["word"] for t in new_tokens]
        variant_sentence = " ".join(words)

        # Record token order
        order = [t["id"] for t in new_tokens]

        variants.append({
            "sentence": variant_sentence,
            "order": order
        })

        if len(variants) >= max_variants:
            break

    return variants


# ------------------------------------------------------------
# Function: build_variant_dataset
# Purpose:
#   Builds dataset of (reference, variant) sentence pairs.
#   Each reference sentence is paired with its generated
#   variants.
# ------------------------------------------------------------
def build_variant_dataset(sentences):

    dataset = []

    for i, sentence in enumerate(sentences):

        variants = generate_variants_subtrees(sentence)

        if len(variants) <= 1:
            continue

        reference = " ".join([t["word"] for t in sentence])

        # skip the original order
        for v in variants[1:]:

            dataset.append({
                "sentence_id": i,
                "tokens": sentence,
                "reference": reference,
                "variant": v["sentence"],
                "order": v["order"]
            })

    return dataset


# ------------------------------------------------------------
# MAIN SCRIPT
# Demonstrates the full pipeline:
#   - Load treebank
#   - Filter valid sentences
#   - Generate permutations
#   - Build dataset
# ------------------------------------------------------------
if __name__ == "__main__":

    sentences = load_conllu(TREEBANK_PATH)

    valid_sentences = [s for s in sentences if is_valid_treebank_sentence(s)]

    print("Total sentences:", len(sentences))
    print("Valid treebank sentences:", len(valid_sentences))

    example = valid_sentences[0]

    print("\nExample sentence:\n")

    for token in example:
        print(
            token["id"],
            token["word"],
            token["upos"],
            "HEAD:", token["head"],
            "REL:", token["deprel"]
        )

    root_id = None

    for token in example:
        if token["head"] == 0:
            root_id = token["id"]
            break 

    print("\nRoot:", root_id)

    print("\nRoot dependents:")

    for token in example:

        if token["head"] == root_id:

            phrase = subtree_tokens(example, token["id"])

            words = [t["word"] for t in phrase]

            print(token["deprel"], "->", " ".join(words))

    variants = generate_variants_subtrees(example)

    print("\nGenerated variants:")

    for v in variants:
        print(v)

    dataset = build_variant_dataset(valid_sentences)

    print("\nTotal variant pairs:", len(dataset))

    print("\nRaw variant repr:")
    print(repr(dataset[0]["variant"]))
    print("\nExample pair:")
    pair = dataset[0]

    print("\nSentence ID:", pair["sentence_id"])

    print("\nReference:")
    print(pair["reference"])

    print("\nVariant:")
    print(pair["variant"])
