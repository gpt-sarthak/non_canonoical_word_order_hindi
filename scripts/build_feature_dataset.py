"""
build_feature_dataset.py

Full feature extraction pipeline following:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

Steps:
    1.  Load HDTB treebank
    2.  Filter sentences (paper criteria)
    3.  Generate permuted variants (max 99, random sample, deprel filter)
    4.  Dependency Length (DL)
    5.  Trigram surprisal  (NLTK MLE, three-level backoff)
    6.  LSTM surprisal
    7.  Adaptive LSTM surprisal  (adapt on context, reset per sentence)
    8.  Information Status / givenness score
    9.  Save features.csv

IS features (Step 8) require token-level parse data (tokens, order,
context) which exist in memory during the pipeline but are not saved
to CSV. They MUST be computed before Step 9.
"""

import os
import pickle
import pandas as pd

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset,
)

from feature_extraction.dl_features       import compute_dl_features
from feature_extraction.trigram_features  import compute_trigram_features
from feature_extraction.lstm_features     import (
    load_lstm_model,
    load_vocab,
    compute_lstm_features,
)
from feature_extraction.adaptive_features import compute_adaptive_features
from feature_extraction.is_features       import compute_is_features


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
TRIGRAM_PATH  = "models/trigram/trigram.pkl"
LSTM_PATH     = "models/lstm/base_model.pt"
VOCAB_PATH    = "data/processed/vocab.pkl"
OUTPUT_PATH   = "data/features/features.csv"

# Columns to drop before saving — internal fields not needed
# for the ranking model
DROP_COLS = ["tokens", "order", "sent_id", "context"]


def main():

    # ── Step 1: Load treebank ──────────────────────────────────
    print("=" * 50)
    print("Step 1: Loading treebank...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)
    print(f"  Sentences loaded : {len(sentences)}")

    # ── Step 2: Filter sentences ───────────────────────────────
    print("\nStep 2: Filtering sentences...")
    valid_triples = [
        (s, sid, ctx)
        for s, sid, ctx in zip(sentences, sent_ids, contexts)
        if is_valid_treebank_sentence(s)
    ]
    vs  = [x[0] for x in valid_triples]
    vsi = [x[1] for x in valid_triples]
    vc  = [x[2] for x in valid_triples]
    print(f"  Valid sentences  : {len(vs)}")
    print(f"  Dropped          : {len(sentences) - len(vs)}")

    # ── Step 3: Generate variants ──────────────────────────────
    print("\nStep 3: Generating variants...")
    dataset = build_variant_dataset(vs, vsi, vc)
    print(f"  Variant pairs    : {len(dataset)}")

    from collections import Counter
    dist = Counter(d["construction_type"] for d in dataset)
    for ctype, n in sorted(dist.items()):
        print(f"    {ctype:<8}: {n:>6}  ({n/len(dataset)*100:.1f}%)")

    # ── Step 4: Dependency Length ──────────────────────────────
    print("\nStep 4: Computing dependency length features...")
    results = compute_dl_features(dataset)
    print(f"  Done — {len(results)} pairs")

    # ── Step 5: Trigram surprisal ──────────────────────────────
    print("\nStep 5: Loading trigram model...")
    with open(TRIGRAM_PATH, "rb") as f:
        trigram_model = pickle.load(f)
    print(f"  Model type : {type(trigram_model).__name__}")
    print("  Computing trigram surprisal...")
    results = compute_trigram_features(results, trigram_model)
    print(f"  Done — sample delta_trigram : {results[0]['delta_trigram']:.4f}")

    # ── Step 6: LSTM surprisal ─────────────────────────────────
    print("\nStep 6: Loading LSTM model and vocabulary...")
    vocab        = load_vocab(VOCAB_PATH)
    lstm, device = load_lstm_model(LSTM_PATH, len(vocab["word2idx"]))
    print(f"  Vocab size : {len(vocab['word2idx'])}")
    print(f"  Device     : {device}")
    print("  Computing LSTM surprisal...")
    results = compute_lstm_features(results, lstm, vocab["word2idx"], device)
    print(f"  Done — sample delta_lstm : {results[0]['delta_lstm']:.4f}")

    # ── Step 7: Adaptive surprisal ─────────────────────────────
    print("\nStep 7: Computing adaptive surprisal...")
    print("  (slowest step — resets model weights per unique sentence)")
    results = compute_adaptive_features(results, lstm, vocab["word2idx"], device)
    print(f"  Done — sample delta_adaptive : {results[0]['delta_adaptive']:.4f}")

    # ── Step 8: Information Status (givenness) ─────────────────
    # Must be computed BEFORE saving because it needs:
    #   item["tokens"] — original dependency parse
    #   item["order"]  — variant token order
    #   item["context"] — preceding sentence string
    # All three are dropped from the CSV after this step.
    print("\nStep 8: Computing Information Status (givenness) features...")
    results = compute_is_features(results)

    # Sanity check IS distribution
    from collections import Counter as C
    is_dist = C(r["is_reference"] for r in results)
    print(f"  IS reference distribution: {dict(sorted(is_dist.items()))}")
    print(f"  Done — sample delta_is : {results[0]['delta_is']}")

    # ── Step 9: Save ───────────────────────────────────────────
    print("\nStep 9: Saving feature dataset...")
    df = pd.DataFrame(results).drop(columns=DROP_COLS, errors="ignore")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"  Saved to  : {OUTPUT_PATH}")
    print(f"  Shape     : {df.shape}")

    print(f"\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    print(f"\nConstruction type distribution:")
    print(df["construction_type"].value_counts().to_string())

    print(f"\nIS score distribution (reference sentences):")
    print(df["is_reference"].value_counts().sort_index().to_string())

    print(f"\nSample deltas (first 5 rows):")
    print(df[["construction_type", "delta_dl", "delta_trigram",
              "delta_lstm", "delta_adaptive", "delta_is"]].head(5).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()