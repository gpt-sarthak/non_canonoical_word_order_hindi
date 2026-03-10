from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    build_variant_dataset
)

from feature_extraction.dl_features import compute_dl_features

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"


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

    print("Computing dependency length features...")
    results = compute_dl_features(dataset)

    print("\nExample result:")
    example = results[0]

    for k, v in example.items():
        print(k, ":", v)
    
    
    better = 0

    for r in results:
        if r["dl_reference"] <= r["dl_variant"]:
            better += 1

    print("Reference shorter DL ratio:", better / len(results))


if __name__ == "__main__":
    run_test()