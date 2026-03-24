"""
test_trigram.py

Tests trigram surprisal feature extraction and reports backoff distribution.

Output saved to: tests/output_test/trigram_test.txt
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset,
)
from feature_extraction.dl_features import compute_dl_features
from feature_extraction.trigram_features import (
    load_trigram_model,
    compute_trigram_features,
    per_word_trigram_surprisal,
)

TREEBANK_PATH  = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
TRIGRAM_PATH   = "models/trigram/trigram.pkl"
OUTPUT_PATH    = "tests/output_test/trigram_test.txt"

# Number of sentences to sample for per-word backoff analysis
BACKOFF_SAMPLE = 200


def run_test():

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("Loading treebank...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)
    log(f"  Total sentences: {len(sentences)}")

    log("Filtering sentences...")
    valid_sentences = [s for s in sentences if is_valid_treebank_sentence(s)]
    log(f"  Valid sentences: {len(valid_sentences)}")

    log("Generating variants...")
    dataset = build_variant_dataset(valid_sentences)
    log(f"  Total pairs: {len(dataset)}")

    log("\nComputing dependency length features...")
    results = compute_dl_features(dataset)

    log("\nLoading trigram model...")
    trigram_model = load_trigram_model(TRIGRAM_PATH)
    log(f"  Model type: {type(trigram_model).__name__}")

    log("Computing trigram surprisal features...")
    results = compute_trigram_features(results, trigram_model)
    log(f"  Done — {len(results)} pairs")

    # ── Example result ─────────────────────────────────────────
    example = results[0]
    log("\nExample result:")
    log(f"  sentence_id      : {example['sentence_id']}")
    log(f"  reference        : {example['reference']}")
    log(f"  variant          : {example['variant']}")
    log(f"  trigram_reference: {example['trigram_reference']:.4f}")
    log(f"  trigram_variant  : {example['trigram_variant']:.4f}")
    log(f"  delta_trigram    : {example['delta_trigram']:.4f}")

    # ── Delta distribution ──────────────────────────────────────
    deltas = [r["delta_trigram"] for r in results]
    neg  = sum(1 for d in deltas if d < 0)
    pos  = sum(1 for d in deltas if d > 0)
    zer  = sum(1 for d in deltas if d == 0)
    mean = sum(deltas) / len(deltas)

    log("\nDelta distribution (delta = ref - var):")
    log(f"  Mean delta_trigram: {mean:.4f}")
    log(f"  Negative (ref more predictable): {neg:>6}  ({neg/len(deltas)*100:.1f}%)")
    log(f"  Zero                           : {zer:>6}  ({zer/len(deltas)*100:.1f}%)")
    log(f"  Positive (var more predictable): {pos:>6}  ({pos/len(deltas)*100:.1f}%)")

    # ── Backoff distribution ────────────────────────────────────
    log(f"\nPer-word backoff distribution (first {BACKOFF_SAMPLE} reference sentences):")

    seen_refs = []
    seen_ids  = set()
    for r in results:
        sid = r["sentence_id"]
        if sid not in seen_ids:
            seen_ids.add(sid)
            seen_refs.append(r["reference"])
        if len(seen_refs) >= BACKOFF_SAMPLE:
            break

    backoff_counts = Counter()
    total_words    = 0

    for sent in seen_refs:
        word_data = per_word_trigram_surprisal(sent, trigram_model)
        for w in word_data:
            if w["backoff"] != "no context":
                backoff_counts[w["backoff"]] += 1
                total_words += 1

    if total_words > 0:
        for level in ["trigram", "bigram", "unigram", "epsilon (OOV)"]:
            n = backoff_counts.get(level, 0)
            log(f"  {level:<20}: {n:>6}  ({n/total_words*100:.1f}%)")

    log(f"\n  Total words analysed: {total_words}")

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_test()
