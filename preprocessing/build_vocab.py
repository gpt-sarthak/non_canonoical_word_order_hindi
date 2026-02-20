import os
from collections import Counter
from tqdm import tqdm
import pickle

INPUT_FILE = "data/processed/wiki_sentences.txt"
OUTPUT_FILE = "data/processed/vocab.pkl"

VOCAB_SIZE = 30000

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

def build_vocab():
    counter = Counter()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            tokens = line.strip().split()
            counter.update(tokens)

    most_common = counter.most_common(VOCAB_SIZE - len(SPECIAL_TOKENS))
    vocab_words = SPECIAL_TOKENS + [word for word, _ in most_common]

    word2idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    vocab = {
        "word2idx": word2idx,
        "idx2word": idx2word
    }

    os.makedirs("data/processed", exist_ok=True)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Vocabulary size: {len(word2idx)}")
    print("Saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    build_vocab()
