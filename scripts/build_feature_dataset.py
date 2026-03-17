import os
import pickle
import pandas as pd

# ------------------------------------------------------------
# Treebank loading and variant generation
# ------------------------------------------------------------
from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset
)

# ------------------------------------------------------------
# Feature extraction modules
# ------------------------------------------------------------
from feature_extraction.dl_features import compute_dl_features
from feature_extraction.trigram_features import compute_trigram_features

from feature_extraction.lstm_features import (
    load_lstm_model,
    load_vocab,
    compute_lstm_features
)

from feature_extraction.adaptive_features import compute_adaptive_features


# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"

TRIGRAM_PATH = "models/trigram/trigram.pkl"

LSTM_PATH = "models/lstm/base_model.pt"

VOCAB_PATH = "data/processed/vocab.pkl"

OUTPUT_PATH = "data/features/features.csv"


def main():

    # ------------------------------------------------------------
    # Step 1: Load Hindi Dependency Treebank
    # ------------------------------------------------------------
    print("Loading treebank...")

    sentences = load_conllu(TREEBANK_PATH)

    # ------------------------------------------------------------
    # Step 2: Filter sentences according to the paper's criteria
    # ------------------------------------------------------------
    # Criteria implemented in is_valid_treebank_sentence():
    #
    # 1. sentence contains subject and object
    # 2. dependency tree is projective
    # 3. sentence is declarative
    # 4. root node is a finite verb
    # 5. root has at least two preverbal dependents
    print("Filtering sentences...")

    valid_sentences = [
        s for s in sentences if is_valid_treebank_sentence(s)
    ]

    print("Valid sentences:", len(valid_sentences))

    # ------------------------------------------------------------
    # Step 3: Generate sentence variants
    # ------------------------------------------------------------
    # Variants are created by permuting preverbal dependency
    # subtrees while keeping syntactic structure intact.
    print("Generating variants...")

    dataset = build_variant_dataset(valid_sentences)

    print("Variant pairs:", len(dataset))

    # ------------------------------------------------------------
    # Step 4: Dependency Length Feature
    # ------------------------------------------------------------
    # Dependency length measures memory cost of sentence processing.
    print("Computing dependency length features...")

    results = compute_dl_features(dataset)

    # ------------------------------------------------------------
    # Step 5: Load trigram language model
    # ------------------------------------------------------------
    print("Loading trigram model...")

    with open(TRIGRAM_PATH, "rb") as f:
        trigram_model = pickle.load(f)

    # ------------------------------------------------------------
    # Step 6: Trigram Surprisal Feature
    # ------------------------------------------------------------
    # Measures local word predictability.
    print("Computing trigram features...")

    results = compute_trigram_features(results, trigram_model)

    # ------------------------------------------------------------
    # Step 7: Load vocabulary for LSTM model
    # ------------------------------------------------------------
    print("Loading vocabulary...")

    vocab = load_vocab(VOCAB_PATH)

    # ------------------------------------------------------------
    # Step 8: Load pretrained LSTM language model
    # ------------------------------------------------------------
    print("Loading LSTM model...")

    lstm, device = load_lstm_model(LSTM_PATH, len(vocab["word2idx"]))

    # ------------------------------------------------------------
    # Step 9: LSTM Surprisal Feature
    # ------------------------------------------------------------
    # Captures long-distance context predictability.
    print("Computing LSTM features...")

    results = compute_lstm_features(
        results,
        lstm,
        vocab["word2idx"],
        device
    )

    # ------------------------------------------------------------
    # Step 10: Adaptive Surprisal Feature
    # ------------------------------------------------------------
    # The model adapts after each sentence using the reference order.
    # This simulates human adaptation during reading.
    print("Computing adaptive surprisal features...")

    results = compute_adaptive_features(
        results,
        lstm,
        vocab["word2idx"],
        device
    )

    # ------------------------------------------------------------
    # Step 11: Save final feature dataset
    # ------------------------------------------------------------
    print("Saving dataset...")

    df = pd.DataFrame(results)

    os.makedirs("data/features", exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved feature dataset to:", OUTPUT_PATH)

    print("\nDataset verification:")

    print("Dataset shape:", df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
