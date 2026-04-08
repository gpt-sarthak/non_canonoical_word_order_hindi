"""
wiki_extract.py

Extracts plain text from a Hindi Wikipedia XML dump.

Input  : data/raw/hiwiki-latest-pages-articles.xml.bz2
         (downloaded from https://dumps.wikimedia.org/hiwiki/)
Output : data/processed/wiki_plain.txt
         (one Wikipedia article revision per line, wikitext stripped)

Processing:
    - Opens the bz2-compressed XML dump with mwxml
    - Iterates over all pages, takes only the latest revision
    - Strips wikitext markup (templates, links, tables) via mwparserfromhell
    - Writes cleaned plain text, one article per line

Runtime: ~20-40 minutes for the full Hindi Wikipedia dump (~500k articles).
Next step: preprocessing/wiki_sentence_tokenizer.py
"""

import mwxml
import mwparserfromhell
import bz2
import os
from tqdm import tqdm


INPUT_FILE  = "data/raw/hiwiki-latest-pages-articles.xml.bz2"
OUTPUT_FILE = "data/processed/wiki_plain.txt"


def clean_text(wikitext):
    """Strip wikitext markup and return plain unicode text."""
    parsed = mwparserfromhell.parse(wikitext)
    return parsed.strip_code()


def extract_wikipedia():
    """
    Stream the bz2 dump, clean each article, write to OUTPUT_FILE.

    Uses mwxml to parse the MediaWiki XML format without loading
    the entire dump into memory.
    """
    os.makedirs("data/processed", exist_ok=True)

    with bz2.open(INPUT_FILE, mode="rt", encoding="utf-8", errors="ignore") as dump_file:
        dump = mwxml.Dump.from_file(dump_file)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
            for page in tqdm(dump.pages, desc="Extracting pages"):
                for revision in page:
                    if revision.text:
                        clean = clean_text(revision.text)
                        if clean.strip():
                            out_file.write(clean + "\n")
                    break  # only latest revision per page


if __name__ == "__main__":
    # ── Inspection mode — does NOT re-extract ────────────────────
    # If wiki_plain.txt exists: show file size and line count.
    # If missing: print extraction instructions.
    #
    # To extract (~20-40 min):
    #   venv/Scripts/python.exe preprocessing/wiki_extract.py
    #   (call extract_wikipedia() directly instead of this block)
    # ─────────────────────────────────────────────────────────────
    if not os.path.exists(OUTPUT_FILE):
        print(f"Output not found at {OUTPUT_FILE}")
        print("To extract: venv/Scripts/python.exe preprocessing/wiki_extract.py")
    else:
        size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"Output file   : {OUTPUT_FILE}")
        print(f"Size          : {size_mb:.1f} MB")
        print(f"Lines (pages) : {line_count:,}")
