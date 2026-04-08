"""
build_vocab.py

Builds a fixed-size vocabulary from Hindi Wikipedia sentences.

Input  : data/processed/wiki_sentences.txt  (from wiki_sentence_tokenizer.py)
Output : data/processed/vocab.pkl
         A dict with two keys:
             "word2idx" : {word: int}   — maps surface form → token ID
             "idx2word" : {int: word}   — reverse mapping

Vocabulary:
    - Size capped at VOCAB_SIZE = 30,000 tokens
    - 4 special tokens prepended: <PAD>(0), <UNK>(1), <SOS>(2), <EOS>(3)
    - Remaining slots filled with the most frequent words in the corpus
    - OOV words at inference time are mapped to <UNK> (index 1)

Runtime: ~2-5 minutes for 1M sentences.
Next step: models/trigram/train_trigram.py  OR  models/lstm/train_base_model.py
"""

import os
from collections import Counter
from tqdm import tqdm
import pickle


INPUT_FILE  = "data/processed/wiki_sentences.txt"
OUTPUT_FILE = "data/processed/vocab.pkl"

VOCAB_SIZE     = 30000
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def build_vocab():
    """
    Count word frequencies in wiki_sentences.txt and build vocab.pkl.

    Steps:
        1. Count every token in the corpus
        2. Take the top (VOCAB_SIZE - 4) most frequent words
        3. Prepend the 4 special tokens
        4. Build word2idx and idx2word mappings
        5. Pickle and save
    """
    counter = Counter()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting tokens"):
            tokens = line.strip().split()
            counter.update(tokens)

    # Keep top N regular words (special tokens fill the first 4 slots)
    most_common = counter.most_common(VOCAB_SIZE - len(SPECIAL_TOKENS))
    vocab_words = SPECIAL_TOKENS + [word for word, _ in most_common]

    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    vocab = {"word2idx": word2idx, "idx2word": idx2word}

    os.makedirs("data/processed", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary size : {len(word2idx):,}")
    print(f"Saved to        : {OUTPUT_FILE}")


if __name__ == "__main__":
    # ── Inspection mode — does NOT rebuild ───────────────────────
    # If vocab.pkl exists: load and print stats + top words.
    # If missing: print build instructions.
    #
    # To build (~2-5 min):
    #   venv/Scripts/python.exe preprocessing/build_vocab.py
    #   (call build_vocab() directly instead of this block)
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists(OUTPUT_FILE):
        print(f"Vocab not found at {OUTPUT_FILE}")
        print("To build: venv/Scripts/python.exe preprocessing/build_vocab.py")
    else:
        with open(OUTPUT_FILE, "rb") as f:
            vocab = pickle.load(f)
        w2i = vocab["word2idx"]
        print(f"Vocab file    : {OUTPUT_FILE}")
        print(f"Vocab size    : {len(w2i):,}")
        print(f"Special tokens: {SPECIAL_TOKENS} → IDs {[w2i[t] for t in SPECIAL_TOKENS]}")
        # Show 10 most common regular words (those with lowest IDs after specials)
        regular = [(w, i) for w, i in w2i.items() if w not in SPECIAL_TOKENS]
        top10 = sorted(regular, key=lambda x: x[1])[:10]
        print(f"Top 10 words  : {[w for w, _ in top10]}")
