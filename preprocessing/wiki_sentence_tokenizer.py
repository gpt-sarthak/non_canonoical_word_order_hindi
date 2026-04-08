"""
wiki_sentence_tokenizer.py

Splits Hindi Wikipedia plain text into clean, filtered sentences.

Input  : data/processed/wiki_plain.txt      (from wiki_extract.py)
Output : data/processed/wiki_sentences.txt
         One sentence per line, Devanagari only, 5–50 tokens.

Processing per line (Wikipedia article):
    1. Split on Hindi sentence boundaries: । ! ?
    2. Strip non-Devanagari characters (Latin, digits, punctuation)
    3. Collapse whitespace
    4. Keep only sentences with 5–50 tokens (MIN_TOKENS / MAX_TOKENS)

The resulting file is used to train both the trigram model and the LSTM.

Runtime: ~5-10 minutes for the full Hindi Wikipedia.
Next step: preprocessing/build_vocab.py
"""

import os
import re
from tqdm import tqdm


INPUT_FILE  = "data/processed/wiki_plain.txt"
OUTPUT_FILE = "data/processed/wiki_sentences.txt"

MIN_TOKENS = 5
MAX_TOKENS = 50

# Hindi sentence boundary markers (Devanagari danda + common punctuation)
SENTENCE_SPLIT_REGEX = r"[।!?]"

# Keep only Devanagari Unicode block (U+0900–U+097F) + whitespace + , -
# Everything else (Latin, digits, URLs, brackets) is replaced with a space
DEVANAGARI_REGEX = re.compile(r"[^\u0900-\u097F\s,\-]")


def clean_line(line):
    """Strip non-Devanagari characters and normalise whitespace."""
    line = line.strip()
    line = DEVANAGARI_REGEX.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def tokenize_and_filter(sentence):
    """
    Return the sentence string if it has MIN_TOKENS–MAX_TOKENS tokens,
    else return None (sentence is too short or too long to be useful).
    """
    tokens = sentence.split()
    if MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        return " ".join(tokens)
    return None


def process_file():
    """Split, clean and filter all sentences from wiki_plain.txt."""
    os.makedirs("data/processed", exist_ok=True)

    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Tokenising"):
            # Split on sentence boundaries BEFORE cleaning so that
            # ।, !, ? are still present as delimiters at split time
            sentences = re.split(SENTENCE_SPLIT_REGEX, line)

            for sentence in sentences:
                sentence = clean_line(sentence)
                if not sentence:
                    continue
                processed = tokenize_and_filter(sentence)
                if processed:
                    outfile.write(processed + "\n")
                    count += 1

    print(f"Sentences written: {count:,}")


if __name__ == "__main__":
    # ── Inspection mode — does NOT re-tokenise ───────────────────
    # If wiki_sentences.txt exists: show file stats + sample lines.
    # If missing: print instructions.
    #
    # To tokenise (~5-10 min):
    #   venv/Scripts/python.exe preprocessing/wiki_sentence_tokenizer.py
    #   (call process_file() directly instead of this block)
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists(OUTPUT_FILE):
        print(f"Output not found at {OUTPUT_FILE}")
        print("To tokenise: venv/Scripts/python.exe preprocessing/wiki_sentence_tokenizer.py")
    else:
        size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lengths = [len(l.strip().split()) for l in lines]
        print(f"Output file     : {OUTPUT_FILE}")
        print(f"Size            : {size_mb:.1f} MB")
        print(f"Sentence count  : {len(lines):,}")
        print(f"Avg tokens/sent : {sum(lengths)/len(lengths):.1f}")
        print(f"\nSample sentences:")
        for line in lines[:3]:
            print(f"  {line.strip()}")
