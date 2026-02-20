import os
import re
from tqdm import tqdm

INPUT_FILE = "data/processed/wiki_plain.txt"
OUTPUT_FILE = "data/processed/wiki_sentences.txt"

MIN_TOKENS = 5
MAX_TOKENS = 50

# Hindi sentence boundary markers
SENTENCE_SPLIT_REGEX = r"[।!?]"

# Keep only Devanagari characters + basic punctuation
DEVANAGARI_REGEX = re.compile(r"[^\u0900-\u097F\s]")

def clean_line(line):
    line = line.strip()
    line = DEVANAGARI_REGEX.sub(" ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()

def tokenize_and_filter(sentence):
    tokens = sentence.split()
    if MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        return " ".join(tokens)
    return None

def process_file():
    os.makedirs("data/processed", exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile):
            line = clean_line(line)
            if not line:
                continue

            sentences = re.split(SENTENCE_SPLIT_REGEX, line)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                processed = tokenize_and_filter(sentence)
                if processed:
                    outfile.write(processed + "\n")

if __name__ == "__main__":
    process_file()
