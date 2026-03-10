import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset
)

from feature_extraction.dl_features import compute_dl_features

from feature_extraction.trigram_features import (
    load_trigram_model,
    compute_trigram_features
)

from feature_extraction.lstm_features import (
    load_lstm_model,
    compute_lstm_features,
    load_vocab
)

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
TRIGRAM_PATH = "models/trigram/trigram.pkl"
LSTM_PATH = "models/lstm/base_model.pt"
VOCAB_PATH = "data/processed/vocab.pkl"

def run_test():

    print("Loading treebank...")
    sentences = load_conllu(TREEBANK_PATH)

    print("Filtering sentences...")
    valid = [s for s in sentences if is_valid_treebank_sentence(s)]

    print("Generating variants...")
    dataset = build_variant_dataset(valid)

    print("Total pairs:", len(dataset))

    print("\nComputing dependency length...")
    results = compute_dl_features(dataset)

    print("\nLoading trigram model...")
    trigram = load_trigram_model(TRIGRAM_PATH)

    print("Computing trigram features...")
    results = compute_trigram_features(results, trigram)

    print("\nLoading vocabulary...")
    vocab = load_vocab(VOCAB_PATH)

    # print(list(vocab["word2idx"].keys())[:20])
    # print("UNK present:", "<unk>" in vocab["word2idx"])
    # print("UNK present:", "<UNK>" in vocab["word2idx"])
    # print("UNK present:", "UNK" in vocab["word2idx"])

    print("\nLoading LSTM model...")
    lstm, device = load_lstm_model(LSTM_PATH, len(vocab["word2idx"]))

    print("Computing LSTM features...")
    results = compute_lstm_features(results, lstm, vocab, device)

    example = results[0]

    print("\nExample result:\n")

    print("reference:", example["reference"])
    print("variant:", example["variant"])

    print("\nDL:", example["delta_dl"])
    print("Trigram:", example["delta_trigram"])
    print("LSTM:", example["delta_lstm"])


if __name__ == "__main__":
    run_test()