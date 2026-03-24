"""
test_lstm.py

Tests LSTM surprisal feature extraction end-to-end.

Output saved to: tests/output_test/lstm_test.txt
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
from feature_extraction.trigram_features import load_trigram_model, compute_trigram_features
from feature_extraction.lstm_features import load_lstm_model, compute_lstm_features, load_vocab

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
TRIGRAM_PATH  = "models/trigram/trigram.pkl"
LSTM_PATH     = "models/lstm/base_model.pt"
VOCAB_PATH    = "data/processed/vocab.pkl"
OUTPUT_PATH   = "tests/output_test/lstm_test.txt"


def run_test():

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("Loading treebank...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)
    log(f"  Total sentences: {len(sentences)}")

    log("Filtering sentences...")
    valid = [s for s in sentences if is_valid_treebank_sentence(s)]
    log(f"  Valid sentences: {len(valid)}")

    log("Generating variants...")
    dataset = build_variant_dataset(valid)
    log(f"  Total pairs: {len(dataset)}")

    log("\nComputing dependency length features...")
    results = compute_dl_features(dataset)

    log("\nLoading trigram model...")
    trigram = load_trigram_model(TRIGRAM_PATH)
    log("Computing trigram features...")
    results = compute_trigram_features(results, trigram)

    log("\nLoading vocabulary...")
    vocab = load_vocab(VOCAB_PATH)
    log(f"  Vocab size: {len(vocab['word2idx'])}")

    log("\nLoading LSTM model...")
    lstm, device = load_lstm_model(LSTM_PATH, len(vocab["word2idx"]))
    log(f"  Device: {device}")

    log("Computing LSTM features...")
    results = compute_lstm_features(results, lstm, vocab["word2idx"], device)
    log(f"  Done — {len(results)} pairs")

    example = results[0]
    log("\nExample result:")
    log(f"  reference    : {example['reference']}")
    log(f"  variant      : {example['variant']}")
    log(f"  delta_dl     : {example['delta_dl']:.4f}")
    log(f"  delta_trigram: {example['delta_trigram']:.4f}")
    log(f"  delta_lstm   : {example['delta_lstm']:.4f}")

    # Delta summary
    deltas = [r["delta_lstm"] for r in results]
    neg  = sum(1 for d in deltas if d < 0)
    pos  = sum(1 for d in deltas if d > 0)
    mean = sum(deltas) / len(deltas)

    log("\nDelta summary (delta = ref - var):")
    log(f"  Mean delta_lstm: {mean:.4f}")
    log(f"  Negative (ref more predictable): {neg:>6}  ({neg/len(deltas)*100:.1f}%)")
    log(f"  Positive (var more predictable): {pos:>6}  ({pos/len(deltas)*100:.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_test()
