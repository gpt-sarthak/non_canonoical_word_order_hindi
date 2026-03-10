import sys
import os

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------
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


TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"
TRIGRAM_MODEL_PATH = "models/trigram/trigram.pkl"


def run_test():

    print("Loading treebank...")
    sentences = load_conllu(TREEBANK_PATH)

    print("Filtering sentences...")
    valid_sentences = [
        s for s in sentences if is_valid_treebank_sentence(s)
    ]

    print("Generating variants...")
    dataset = build_variant_dataset(valid_sentences)

    print("Total pairs:", len(dataset))

    # ------------------------------------------------------------
    # Dependency Length
    # ------------------------------------------------------------
    print("\nComputing dependency length features...")
    results = compute_dl_features(dataset)

    # ------------------------------------------------------------
    # Load trigram model
    # ------------------------------------------------------------
    print("\nLoading trigram model...")
    trigram_model = load_trigram_model(TRIGRAM_MODEL_PATH)

    # ------------------------------------------------------------
    # Trigram surprisal
    # ------------------------------------------------------------
    print("Computing trigram surprisal features...")
    results = compute_trigram_features(results, trigram_model)

    # ------------------------------------------------------------
    # Print example result
    # ------------------------------------------------------------
    example = results[0]

    print("\nExample result:\n")

    print("sentence_id :", example["sentence_id"])
    print("reference   :", example["reference"])
    print("variant     :", example["variant"])

    print("\nDependency Length:")
    print("dl_reference:", example["dl_reference"])
    print("dl_variant  :", example["dl_variant"])
    print("delta_dl    :", example["delta_dl"])

    print("\nTrigram Surprisal:")
    print("trigram_reference:", example["trigram_reference"])
    print("trigram_variant  :", example["trigram_variant"])
    print("delta_trigram    :", example["delta_trigram"])


if __name__ == "__main__":
    run_test()