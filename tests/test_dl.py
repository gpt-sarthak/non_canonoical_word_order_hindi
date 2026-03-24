"""
test_dl.py

Tests dependency length feature extraction on the first N pairs
from the HDTB treebank.

Output saved to: tests/output_test/dl_test.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset,
)
from feature_extraction.dl_features import compute_dl_features

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
OUTPUT_PATH   = "tests/output_test/dl_test.txt"


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
    log(f"  Done — {len(results)} pairs")

    log("\nExample result:")
    example = results[0]
    log(f"  sentence_id  : {example['sentence_id']}")
    log(f"  reference    : {example['reference']}")
    log(f"  variant      : {example['variant']}")
    log(f"  dl_reference : {example['dl_reference']}")
    log(f"  dl_variant   : {example['dl_variant']}")
    log(f"  delta_dl     : {example['delta_dl']}")

    # Summary stats
    deltas = [r["delta_dl"] for r in results]
    neg = sum(1 for d in deltas if d < 0)
    pos = sum(1 for d in deltas if d > 0)
    zer = sum(1 for d in deltas if d == 0)
    mean_delta = sum(deltas) / len(deltas)

    log("\nDelta summary (delta = ref - var):")
    log(f"  Mean delta_dl : {mean_delta:.4f}")
    log(f"  Negative (ref shorter DL) : {neg:>6}  ({neg/len(deltas)*100:.1f}%)")
    log(f"  Zero                      : {zer:>6}  ({zer/len(deltas)*100:.1f}%)")
    log(f"  Positive (ref longer DL)  : {pos:>6}  ({pos/len(deltas)*100:.1f}%)")

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_test()
