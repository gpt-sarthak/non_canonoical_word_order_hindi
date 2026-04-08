"""
dl_features.py

Applies dependency length computation to the full (reference, variant) dataset.

Input  (dataset):
    List of pair dicts from hutb_loader / build_variant_dataset, each with:
        "tokens"  — original token list (dependency parse)
        "order"   — list of token IDs in the variant's surface order
        + all other metadata fields (sentence_id, construction_type, etc.)

Output (each item extended with):
    "dl_reference" — total dependency length of the reference word order
    "dl_variant"   — total dependency length of the variant word order
    "delta_dl"     — dl_reference − dl_variant
                     Negative = reference minimises DL (paper predicts this)
                     Positive = variant has shorter dependencies

Delta convention:
    delta = reference − variant  (matches all other delta columns in the paper).
    A negative delta_dl means the reference sentence has fewer/shorter
    dependencies than the scrambled variant — consistent with DLT predictions.
"""

from features.dependency_length import (
    extract_dependencies,
    compute_dependency_length
)


# ------------------------------------------------------------
# Function: compute_dl_features
#
# Purpose:
#   Computes dependency length features for each
#   reference–variant sentence pair.
#
# Input:
#   dataset = [
#       {
#           "sentence_id": int,
#           "tokens": dependency tree tokens,
#           "reference": str,
#           "variant": str,
#           "order": list[int]
#       }
#   ]
#
# Output:
#   List of dictionaries containing DL features.
#
# Each output entry contains:
#   reference sentence
#   variant sentence
#   DL(reference)
#   DL(variant)
#   ΔDL
# ------------------------------------------------------------
def compute_dl_features(dataset):

    results = []

    for item in dataset:

        tokens = item["tokens"]

        # Extract dependency arcs once
        dependencies = extract_dependencies(tokens)

        # Original order of tokens in the treebank sentence
        reference_order = [
            t["id"] for t in tokens
        ]

        # Word order for generated variant
        variant_order = item["order"]

        # Compute dependency lengths
        dl_ref = compute_dependency_length(
            reference_order,
            dependencies
        )

        dl_var = compute_dependency_length(
            variant_order,
            dependencies
        )

        # Paper: delta = feature(reference) − feature(variant)
        # Positive delta means reference has HIGHER DL than variant.
        # Negative delta means reference MINIMISES dependency length.
        delta_dl = dl_ref - dl_var

        # Use **item to carry ALL fields forward (sentence_id,
        # construction_type, context, tokens, etc.) so downstream
        # steps don't lose metadata.
        results.append({
            **item,
            "dl_reference": dl_ref,
            "dl_variant":   dl_var,
            "delta_dl":     delta_dl,
        })

    return results


# ─────────────────────────────────────────────────────────────
# __main__ — standalone debug using real data from features.csv
#
# Loads the first DOSV pair from features.csv and recomputes
# delta_dl from scratch, then compares against the stored value.
#
# Expected (from features.csv, sentence_id=0, DOSV):
#   reference  : इसे नवाब शाहजेहन ने बनवाया था ।
#   variant    : नवाब शाहजेहन ने इसे बनवाया था ।
#   delta_dl   : 2
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from data.hutb_loader import load_conllu, is_valid_treebank_sentence, build_variant_dataset
    import pandas as pd

    # Load stored delta to verify against
    df = pd.read_csv("data/features/features.csv")
    row = df[df["construction_type"] == "DOSV"].iloc[0]
    expected_delta = row["delta_dl"]
    target_ref     = row["reference"]
    target_var     = row["variant"]

    print(f"Target pair from features.csv:")
    print(f"  reference  : {target_ref}")
    print(f"  variant    : {target_var}")
    print(f"  stored delta_dl : {expected_delta}")

    # Re-derive from treebank
    print("\nRecomputing from treebank...")
    sentences, sent_ids, contexts = load_conllu("data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu")
    valid = [(s, sid, ctx) for s, sid, ctx in zip(sentences, sent_ids, contexts)
             if is_valid_treebank_sentence(s)]
    first_s, first_sid, first_ctx = valid[0]
    dataset = build_variant_dataset([first_s], [first_sid], [first_ctx])

    # Find the specific variant that matches
    match = next((d for d in dataset if d["variant"] == target_var), dataset[0])
    results = compute_dl_features([match])
    r = results[0]

    print(f"  recomputed delta_dl : {r['delta_dl']}")
    print(f"  match stored value  : {r['delta_dl'] == expected_delta}")