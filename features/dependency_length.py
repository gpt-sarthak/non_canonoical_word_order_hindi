"""
Dependency Length Feature Implementation

This module implements the Dependency Length (DL) metric used in the paper.
Dependency length is based on Dependency Locality Theory and is defined as:

DL = Σ |position(word) − position(head)|

Where:
- position(word) is the position of a dependent in the sentence
- position(head) is the position of its syntactic head

The implementation is optimized by extracting dependency arcs once
and reusing them across multiple sentence variants.
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


