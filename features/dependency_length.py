"""
dependency_length.py

Implements the Dependency Length (DL) metric from Dependency Locality Theory.

Paper (Gibson 2000, as used in Ranjan & van Schijndel 2024):
    DL = Σ |position(dependent) − position(head)|

    Shorter dependency length = adjacent words are syntactically related,
    which reduces memory load during incremental parsing.

Design:
    Dependency arcs (dependent_id, head_id) are extracted ONCE from the
    original treebank token list, then reused across all permuted variants
    of that sentence — avoiding redundant re-parsing.

Input (to compute_dependency_length):
    order        — list of token IDs in surface word order (e.g. [3,1,2,4])
    dependencies — list of (dependent_id, head_id) pairs

Output:
    A single integer: the total dependency length for that word order.
"""


# ------------------------------------------------------------
# Function: extract_dependencies
#
# Purpose:
#   Extracts dependency arcs from a tokenized sentence.
#
# Why:
#   Instead of recomputing dependencies for each variant, we
#   extract the (dependent, head) pairs once and reuse them
#   during dependency length computation.
#
# Output:
#   List of tuples:
#   [(dependent_id, head_id), ...]
# ------------------------------------------------------------
def extract_dependencies(tokens):

    deps = []

    for token in tokens:

        # Skip the root (head == 0)
        if token["head"] == 0:
            continue

        deps.append((token["id"], token["head"]))

    return deps


# ------------------------------------------------------------
# Function: compute_dependency_length
#
# Purpose:
#   Computes dependency length for a given word order.
#
# Inputs:
#   order        -> list of token IDs representing word order
#   dependencies -> list of (dependent_id, head_id) pairs
#
# Example:
#   order = [2,3,4,1,5]
#
#   means:
#   position 1 -> token 2
#   position 2 -> token 3
#   position 3 -> token 4
#   position 4 -> token 1
#   position 5 -> token 5
#
# Steps:
#   1. Convert order list into token_id → position mapping
#   2. Compute |position(dep) − position(head)|
#   3. Sum over all dependencies
#
# Returns:
#   total dependency length
# ------------------------------------------------------------
def compute_dependency_length(order, dependencies):

    # Build mapping: token_id → position in sentence
    position_map = {
        token_id: i + 1
        for i, token_id in enumerate(order)
    }

    total_dl = 0

    for dep_id, head_id in dependencies:

        dep_pos = position_map[dep_id]
        head_pos = position_map[head_id]

        total_dl += abs(dep_pos - head_pos)

    return total_dl


# ─────────────────────────────────────────────────────────────
# __main__ — standalone debug / sanity check
#
# Constructs a synthetic 5-token sentence with known arc structure
# and verifies that DL is computed correctly for two orderings.
#
# Sentence structure (canonical SOV):
#   Token 1: राम   (nsubj → verb at 3)
#   Token 2: किताब (obj   → verb at 3)
#   Token 3: पढ़ता  (root)
#   Token 4: है     (aux   → verb at 3)
#
# Arc set: (1→3), (2→3), (4→3)
#
# Reference order [1,2,3,4]:
#   |1-3| + |2-3| + |4-3| = 2 + 1 + 1 = 4
#
# Scrambled order [2,1,3,4]  (obj before subj):
#   token positions: 2→1, 1→2, 3→3, 4→4
#   |pos(1)-pos(3)| + |pos(2)-pos(3)| + |pos(4)-pos(3)|
#   = |2-3| + |1-3| + |4-3| = 1 + 2 + 1 = 4  (same DL here)
#
# Inverted order [2,3,1,4]  (obj-verb-subj):
#   positions: 2→1, 3→2, 1→3, 4→4
#   |pos(1)-pos(3)| + |pos(2)-pos(3)| + |pos(4)-pos(3)|
#   = |3-2| + |1-2| + |4-2| = 1 + 1 + 2 = 4  (same again — symmetric)
#
# To see DL change: move root away from its dependents.
# Order [1,2,4,3] (aux before verb):
#   positions: 1→1, 2→2, 4→3, 3→4
#   |1-4| + |2-4| + |3-4| = 3 + 2 + 1 = 6  (longer)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Real sentence from features.csv (sentence_id=0, DOSV) ────
    # reference : इसे नवाब शाहजेहन ने बनवाया था ।
    # variant   : नवाब शाहजेहन ने इसे बनवाया था ।
    # Expected delta_dl = +2  (reference has longer dependencies)
    #
    # Rather than hard-coding the full token list here, we load the
    # treebank, filter, generate one pair, and compute DL on it —
    # this also serves as an integration smoke-test for the loader.
    # ─────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, ".")

    from data.hutb_loader import load_conllu, is_valid_treebank_sentence, build_variant_dataset

    print("Loading treebank and generating first valid pair...")
    sentences, sent_ids, contexts = load_conllu("data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu")
    valid = [
        (s, sid, ctx)
        for s, sid, ctx in zip(sentences, sent_ids, contexts)
        if is_valid_treebank_sentence(s)
    ]
    # Use only the first sentence to keep this fast
    first_s, first_sid, first_ctx = valid[0][0], valid[0][1], valid[0][2]
    dataset = build_variant_dataset([first_s], [first_sid], [first_ctx])

    if not dataset:
        print("No variants generated for first sentence — try another.")
        sys.exit(1)

    item = dataset[0]
    deps = extract_dependencies(item["tokens"])
    ref_order = [t["id"] for t in item["tokens"]]
    var_order = item["order"]

    dl_ref = compute_dependency_length(ref_order, deps)
    dl_var = compute_dependency_length(var_order, deps)
    delta  = dl_ref - dl_var

    print(f"reference : {item['reference']}")
    print(f"variant   : {item['variant']}")
    print(f"dl_ref    : {dl_ref}")
    print(f"dl_var    : {dl_var}")
    print(f"delta_dl  : {delta}  (positive = reference has longer deps)")
