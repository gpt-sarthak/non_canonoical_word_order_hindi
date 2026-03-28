"""
add_pcfg_features.py

Patches real PCFG surprisal values into the existing features.csv,
replacing the placeholder log(len+1) values.

This avoids re-running the slow adaptive-LSTM pipeline (~hours).
Runtime: ~2-3 minutes.

Steps:
    1. Load existing features.csv
    2. Reload dataset from hutb_loader (needed for tokens/order per pair)
    3. Extract constituency trees from all HUTB train sentences
    4. Run 5-fold CV to score reference sentences leak-free
    5. Score variants using fast direct chunk scoring
    6. Merge delta_pcfg into features.csv and save
"""

import os
import sys
sys.path.insert(0, ".")
import pandas as pd

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset,
)
from feature_extraction.pcfg_features import (
    extract_trees_from_conllu,
    compute_pcfg_surprisal_cv,
    score_chunks,
    score_variant_from_tokens_and_order,
)

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
FEATURES_PATH = "data/features/features.csv"


def main():
    print("=" * 55)
    print("PCFG Feature Patching")
    print("=" * 55)

    # ── Step 1: Load existing features.csv ────────────────────
    print("\nStep 1: Loading existing features.csv ...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  Rows: {len(df)},  Columns: {list(df.columns)}")

    # ── Step 2: Rebuild in-memory dataset ─────────────────────
    # We need the 'tokens' and 'order' fields per pair (not saved in CSV)
    # to use the fast direct chunk scoring for variants.
    print("\nStep 2: Rebuilding in-memory dataset from HUTB ...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)
    print(f"  Sentences loaded: {len(sentences)}")

    valid_triples = [
        (s, sid, ctx)
        for s, sid, ctx in zip(sentences, sent_ids, contexts)
        if is_valid_treebank_sentence(s)
    ]
    vs  = [x[0] for x in valid_triples]
    vsi = [x[1] for x in valid_triples]
    vc  = [x[2] for x in valid_triples]
    print(f"  Valid sentences:  {len(vs)}")

    dataset = build_variant_dataset(vs, vsi, vc)
    print(f"  Variant pairs:    {len(dataset)}")

    # ── Step 3: Extract constituency trees ────────────────────
    print("\nStep 3: Extracting constituency trees from HUTB ...")
    all_hutb_trees = extract_trees_from_conllu(TREEBANK_PATH)
    print(f"  Trees extracted:  {len(all_hutb_trees)}")

    # ── Step 4: Run 5-fold CV to score references (single CV pass) ───────────
    print("\nStep 4: Computing PCFG surprisal via 5-fold CV ...")
    study_sentences = list({d["reference"] for d in dataset})
    print(f"  Unique reference sentences: {len(study_sentences)}")

    # sent_to_pcfg: {surface → (surprisal, fold_pcfg)}
    # fold_pcfg is the model that MUST also score this sentence's variants
    sent_to_pcfg, full_pcfg = compute_pcfg_surprisal_cv(
        all_hutb_trees, study_sentences, n_folds=5
    )

    # ── Step 5: Score variants using fold PCFG (no CV bias) ───────────────
    print("\nStep 5: Scoring variants with fold PCFGs ...")

    ref_lookup = {}
    var_lookup = {}
    for item in dataset:
        ref_str = item["reference"]
        var_str = item["variant"]

        ref_entry = sent_to_pcfg.get(ref_str, (float("nan"), full_pcfg))
        ref_s, fold_pcfg_item = ref_entry
        ref_lookup[ref_str] = ref_s

        tokens = item.get("tokens", [])
        order  = item.get("order",  [])
        if tokens and order:
            var_s = score_variant_from_tokens_and_order(tokens, order, fold_pcfg_item)
        else:
            var_s = float("nan")
        var_lookup[var_str] = var_s

    print(f"  Unique references scored : {len(ref_lookup)}")
    print(f"  Unique variants scored   : {len(var_lookup)}")

    # ── Step 6: Build lookup and patch features.csv ───────────
    print("\nStep 6: Patching features.csv ...")

    # Build surface → chunk_rules lookup for fallback variant scoring
    tree_by_surface = {t["sentence"]: t for t in all_hutb_trees}

    def score_variant_surface(variant_str, ref_surface):
        """
        Score a variant surface string not in var_lookup.
        Uses the SAME fold_pcfg that scored the reference (no CV bias).
        """
        ref_entry = sent_to_pcfg.get(ref_surface)
        if ref_entry is None:
            return float("nan")
        _, pcfg_to_use = ref_entry

        ref_tree = tree_by_surface.get(ref_surface)
        if ref_tree is None:
            return float("nan")

        ref_chunks = ref_tree["chunk_rules"]   # [(label, [word,...]), ...]
        var_words  = variant_str.lower().split()

        # Tile the variant words using the reference's chunk word-lists.
        chunk_word_to_label = {}
        for lbl, words in ref_chunks:
            chunk_word_to_label[tuple(w.lower() for w in words)] = lbl

        matched_chunks = []
        pos = 0
        n   = len(var_words)
        while pos < n:
            found = False
            for length in range(min(n - pos, max(len(w) for w in chunk_word_to_label)), 0, -1):
                span = tuple(var_words[pos:pos + length])
                if span in chunk_word_to_label:
                    matched_chunks.append((chunk_word_to_label[span], list(span)))
                    pos += length
                    found = True
                    break
            if not found:
                matched_chunks.append(("UNK_CHUNK", [var_words[pos]]))
                pos += 1

        return score_chunks(matched_chunks, pcfg_to_use)

    # Patch the CSV
    matched_ref  = 0
    matched_var  = 0
    fallback_var = 0

    pcfg_ref_col   = []
    pcfg_var_col   = []
    delta_pcfg_col = []

    for _, row in df.iterrows():
        ref_str = row["reference"]
        var_str = row["variant"]

        # Reference: always use CV-computed value
        ref_s = ref_lookup.get(ref_str, float("nan"))
        if ref_s == ref_s:   # not NaN
            matched_ref += 1

        # Variant: use var_lookup if available, else reconstruct with fold PCFG
        if var_str in var_lookup:
            var_s = var_lookup[var_str]
            matched_var += 1
        else:
            var_s = score_variant_surface(var_str, ref_str)
            fallback_var += 1

        pcfg_ref_col.append(ref_s)
        pcfg_var_col.append(var_s)
        delta_pcfg_col.append(ref_s - var_s)

    print(f"  Reference matches (CV)   : {matched_ref}")
    print(f"  Variant matches (direct) : {matched_var}")
    print(f"  Variant fallback (infer) : {fallback_var}")

    df["pcfg_reference"] = pcfg_ref_col
    df["pcfg_variant"]   = pcfg_var_col
    df["delta_pcfg"]     = delta_pcfg_col

    df.to_csv(FEATURES_PATH, index=False)
    print(f"\n  Saved to: {FEATURES_PATH}")
    print(f"  Shape:    {df.shape}")

    # ── Summary ────────────────────────────────────────────────
    print("\nDelta PCFG summary by construction type:")
    for ctype in ["DOSV", "IOSV", "OSV", "SOV"]:
        subset = df[df["construction_type"] == ctype]["delta_pcfg"]
        if len(subset):
            print(f"  {ctype:<6}  n={len(subset):>6}  "
                  f"mean={subset.mean():+.3f}  "
                  f"std={subset.std():.3f}")

    print("\nSample rows (first 5):")
    print(df[["construction_type", "reference", "delta_pcfg"]].head().to_string())

    print("\nDone. Re-run scripts/train_ranking_model.py to see updated results.")


if __name__ == "__main__":
    main()
