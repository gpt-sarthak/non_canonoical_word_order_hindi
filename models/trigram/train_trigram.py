"""
train_trigram.py

Trains an NLTK MLE trigram language model on Hindi Wikipedia sentences
and saves it as a pickle file for use in trigram surprisal scoring.

Input  : data/processed/wiki_sentences.txt  (one sentence per line)
         data/processed/vocab.pkl           (word2idx / idx2word dicts)
Output : models/trigram/trigram.pkl         (pickled NLTK MLE object)

Training details:
    - Uses NLTK's padded_everygram_pipeline to generate 1-, 2-, 3-grams
    - Vocabulary is capped at 30,000 tokens (from build_vocab.py)
    - Max sentences: 1,000,000 (subset of full Wikipedia for speed)
    - No smoothing (MLE) — backoff handled at scoring time in
      trigram_features.py via three-level fallback (tri → bi → uni → ε)

Runtime: ~5-10 minutes depending on available RAM.
"""

import os
import pickle
from tqdm import tqdm
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

INPUT_FILE  = "data/processed/wiki_sentences.txt"
VOCAB_FILE  = "data/processed/vocab.pkl"
OUTPUT_FILE = "models/trigram/trigram.pkl"

MAX_SENTENCES = 1_000_000
N = 3  # trigram


def load_vocab():
    """Load word2idx mapping from vocab.pkl."""
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    return vocab["word2idx"]


def train_trigram():
    """
    Train MLE trigram on Hindi Wikipedia and save to OUTPUT_FILE.

    Steps:
        1. Load vocabulary (30k words) from vocab.pkl
        2. Read up to MAX_SENTENCES from wiki_sentences.txt
        3. Replace OOV tokens with <UNK> to match vocab
        4. Fit NLTK MLE trigram via padded_everygram_pipeline
        5. Pickle and save the model
    """
    word2idx  = load_vocab()
    vocab_set = set(word2idx.keys())

    sentences = []
    count     = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading sentences"):
            tokens = line.strip().split()
            # Replace OOV tokens with <UNK> to stay within the fixed vocabulary
            tokens = [t if t in vocab_set else "<UNK>" for t in tokens]
            sentences.append(tokens)
            count += 1
            if count >= MAX_SENTENCES:
                break

    print(f"  Sentences loaded : {count}")

    # padded_everygram_pipeline generates 1-, 2-, 3-grams with <s>/<\/s> padding
    train_data, padded_vocab = padded_everygram_pipeline(N, sentences)

    model = MLE(N)
    model.fit(train_data, padded_vocab)

    os.makedirs("models/trigram", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Trigram model saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    # ── Inspection mode — does NOT retrain ───────────────────────
    # If trigram.pkl exists: load and print model stats.
    # If missing: print training instructions.
    #
    # To actually train (~5-10 min):
    #   venv/Scripts/python.exe models/trigram/train_trigram.py
    #   (call train_trigram() directly instead of this inspect block)
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists(OUTPUT_FILE):
        print(f"No trained model found at {OUTPUT_FILE}")
        print("To train: venv/Scripts/python.exe models/trigram/train_trigram.py")
    else:
        with open(OUTPUT_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from : {OUTPUT_FILE}")
        print(f"Model type        : {type(model).__name__}")
        print(f"N-gram order      : {model.order}")
        vocab_size = len(model.vocab)
        print(f"Vocab size        : {vocab_size:,}")
        # Sanity check: score a common Hindi word
        test_word = "है"
        prob = model.score(test_word)
        print(f"P('{test_word}') unigram : {prob:.6f}")
