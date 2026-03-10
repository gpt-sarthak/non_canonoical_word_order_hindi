"""
Dependency Length Feature Extraction

This module applies the dependency length computation to
the variant dataset generated from the Hindi Dependency Treebank.

For each (reference, variant) pair we compute:

DL_reference
DL_variant
ΔDL = DL_variant − DL_reference

This difference is later used as a predictor in the
logistic regression ranking model.
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

        # Feature used in ranking model
        delta_dl = dl_var - dl_ref

        results.append({
            "sentence_id": item["sentence_id"],
            "reference": item["reference"],
            "variant": item["variant"],
            "dl_reference": dl_ref,
            "dl_variant": dl_var,
            "delta_dl": delta_dl
        })

    return results