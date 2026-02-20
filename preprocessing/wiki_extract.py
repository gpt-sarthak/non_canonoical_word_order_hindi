import mwxml
import mwparserfromhell
import bz2
import os
from tqdm import tqdm


INPUT_FILE = "data/raw/hiwiki-latest-pages-articles.xml.bz2"
OUTPUT_FILE = "data/processed/wiki_plain.txt"


def clean_text(wikitext):
    parsed = mwparserfromhell.parse(wikitext)
    text = parsed.strip_code()
    return text


def extract_wikipedia():
    os.makedirs("data/processed", exist_ok=True)

    with bz2.open(INPUT_FILE, mode='rt', encoding='utf-8', errors='ignore') as dump_file:
        dump = mwxml.Dump.from_file(dump_file)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
            for page in tqdm(dump.pages):
                for revision in page:
                    if revision.text:
                        clean = clean_text(revision.text)
                        if clean.strip():
                            out_file.write(clean + "\n")
                        break  # only latest revision


if __name__ == "__main__":
    extract_wikipedia()
