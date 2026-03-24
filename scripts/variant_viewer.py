"""
variant_viewer.py

Takes a Hindi sentence from the treebank (by sent_id or sentence index),
generates all valid permuted variants, exports them as CoNLL-U format,
and opens the CoNLL-U viewer at:
    https://urd2.let.rug.nl/~kleiweg/conllu/

The CoNLL-U output contains:
    - The reference sentence (original treebank order)
    - All generated variants with correctly remapped head indices

Head index remapping:
    When a sentence is permuted, token positions change but dependency
    relations do not. We remap each token's head from original_id to
    its new position in the permuted order so the dependency tree is
    still valid in the new linear order.

Usage:
    # By sentence index (0-based among valid sentences)
    python scripts/variant_viewer.py --index 0

    # By sent_id from treebank
    python scripts/variant_viewer.py --sent_id train-s2

    # By Hindi text (partial match against reference sentences)
    python scripts/variant_viewer.py --text "इसे नवाब"

    # Show first N variants only
    python scripts/variant_viewer.py --index 0 --max_show 5

Output:
    output/variants/<sent_id>.conllu   — CoNLL-U file
    Then opens https://urd2.let.rug.nl/~kleiweg/conllu/ in your browser.
    Paste the file contents (or upload the file) to view the trees.
"""

import os
import sys
import argparse
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.hutb_loader import (
    load_conllu,
    is_valid_treebank_sentence,
    generate_variants_subtrees,
    build_corpus_transitions,
    get_construction_type,
    TREEBANK_PATH,
)

VIEWER_URL  = "https://urd2.let.rug.nl/~kleiweg/conllu/"
OUTPUT_DIR  = "output/variants"


# ─────────────────────────────────────────────────────────────
# remap_heads
#
# When tokens are reordered, their CoNLL-U IDs (positions) change.
# The head field must be remapped from original token IDs to the
# new positions in the permuted sentence.
#
# Example:
#   Original order: [1,2,3,4,5]  (token IDs = positions)
#   Variant order:  [3,1,2,4,5]  (token ID 3 is now at position 1)
#
#   position_map = {3:1, 1:2, 2:3, 4:4, 5:5}
#
#   Token originally at id=1 with head=4:
#     new_id   = position_map[1] = 2
#     new_head = position_map[4] = 4
#
# Returns list of (new_id, token_dict, new_head) tuples in new order.
# ─────────────────────────────────────────────────────────────
def remap_heads(tokens_by_id, order):
    """
    Remap token IDs and head indices for a permuted word order.

    tokens_by_id — dict mapping original_id → token dict
    order        — list of original token IDs in the new surface order

    Returns list of token dicts with updated 'id' and 'head' fields,
    in the new surface order.
    """
    # Map: original_id → new_position (1-based)
    position_map = {orig_id: new_pos + 1 for new_pos, orig_id in enumerate(order)}

    remapped = []
    for new_pos, orig_id in enumerate(order):
        tok = dict(tokens_by_id[orig_id])   # shallow copy
        tok["id"]   = new_pos + 1
        orig_head   = tokens_by_id[orig_id]["head"]
        # head=0 means root — keep as 0
        tok["head"] = position_map.get(orig_head, 0) if orig_head != 0 else 0
        remapped.append(tok)

    return remapped


# ─────────────────────────────────────────────────────────────
# tokens_to_conllu
#
# Converts a list of token dicts to CoNLL-U lines.
# Fields not stored (xpos, misc, etc.) are set to '_'.
# ─────────────────────────────────────────────────────────────
def tokens_to_conllu(tokens, sent_id=None, comment=None):
    """
    Render token list as CoNLL-U text block (without trailing newline).

    CoNLL-U columns:
        ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
    """
    lines = []
    if sent_id:
        lines.append(f"# sent_id = {sent_id}")
    if comment:
        lines.append(f"# {comment}")

    # Add text comment (surface string)
    surface = " ".join(t["word"] for t in tokens)
    lines.append(f"# text = {surface}")

    for tok in tokens:
        lines.append(
            "\t".join([
                str(tok["id"]),                         # ID
                tok["word"],                            # FORM
                tok.get("lemma", "_"),                  # LEMMA
                tok.get("upos",  "_"),                  # UPOS
                "_",                                    # XPOS
                tok.get("feats", "_") or "_",           # FEATS
                str(tok["head"]),                       # HEAD
                tok.get("deprel", "_"),                 # DEPREL
                "_",                                    # DEPS
                "_",                                    # MISC
            ])
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# generate_conllu_for_sentence
#
# Given a reference sentence (token list) and its valid variants,
# produce a full CoNLL-U document with the reference first, then
# each variant, each as a separate sentence block.
# ─────────────────────────────────────────────────────────────
def generate_conllu_for_sentence(sentence, variants, sent_id, construction_type):
    """
    Build full CoNLL-U document string for a sentence + all variants.
    """
    tokens_by_id = {t["id"]: t for t in sentence}
    blocks = []

    # ── Reference sentence ──────────────────────────────────
    ref_comment = f"REFERENCE ({construction_type}) | DL={sum(abs(t['id']-t['head']) for t in sentence if t['head']!=0)}"
    block = tokens_to_conllu(sentence, sent_id=f"{sent_id}-REF", comment=ref_comment)
    blocks.append(block)

    # ── Each variant ────────────────────────────────────────
    for i, v in enumerate(variants, 1):
        remapped = remap_heads(tokens_by_id, v["order"])

        # Compute variant DL
        dl_var = sum(
            abs(tok["id"] - tok["head"])
            for tok in remapped
            if tok["head"] != 0
        )

        # Detect construction type of variant (surface order)
        # by checking which deprel comes first among preverbal root-deps
        root_new = next((t for t in remapped if t["head"] == 0), None)
        ctype_var = "unknown"
        if root_new:
            prev = sorted(
                [t for t in remapped if t["head"] == root_new["id"] and t["id"] < root_new["id"]],
                key=lambda t: t["id"]
            )
            if prev:
                first_rel = prev[0]["deprel"]
                if first_rel == "obj":
                    ctype_var = "DOSV"
                elif first_rel == "iobj":
                    ctype_var = "IOSV"
                elif any(t["deprel"] in {"obj","iobj"} for t in prev):
                    ctype_var = "OSV"
                else:
                    ctype_var = "SOV"

        var_comment = f"VARIANT {i:02d} ({ctype_var}) | DL={dl_var}"
        block = tokens_to_conllu(remapped, sent_id=f"{sent_id}-V{i:02d}", comment=var_comment)
        blocks.append(block)

    return "\n\n".join(blocks) + "\n"


# ─────────────────────────────────────────────────────────────
# print_variant_breakdown
#
# Print a summary table of the reference + variants to terminal.
# ─────────────────────────────────────────────────────────────
def print_variant_breakdown(sentence, variants, sent_id, construction_type):
    """Print a readable breakdown of reference + variants to stdout."""

    print(f"\n{'='*65}")
    print(f"Sentence : {sent_id}")
    print(f"Type     : {construction_type}")
    print(f"Variants : {len(variants)}")
    print(f"{'='*65}")

    ref_surface = " ".join(t["word"] for t in sentence)
    ref_dl = sum(abs(t["id"] - t["head"]) for t in sentence if t["head"] != 0)

    print(f"\n  [REF]  DL={ref_dl:<4}  {ref_surface}")

    tokens_by_id = {t["id"]: t for t in sentence}
    for i, v in enumerate(variants, 1):
        remapped = remap_heads(tokens_by_id, v["order"])
        dl_v = sum(abs(t["id"] - t["head"]) for t in remapped if t["head"] != 0)

        # Construction type of variant
        root = next((t for t in remapped if t["head"] == 0), None)
        ctype_v = "?"
        if root:
            prev = sorted(
                [t for t in remapped if t["head"] == root["id"] and t["id"] < root["id"]],
                key=lambda t: t["id"]
            )
            if prev:
                r = prev[0]["deprel"]
                ctype_v = "DOSV" if r=="obj" else "IOSV" if r=="iobj" else "SOV"

        delta = dl_v - ref_dl
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  [V{i:02d}] {ctype_v:<5} DL={dl_v:<4} ({arrow}{abs(delta):<3}) {v['sentence']}")

    print()


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate CoNLL-U variants for a Hindi sentence and open the viewer."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index",   type=int,   help="0-based index among valid sentences")
    group.add_argument("--sent_id", type=str,   help="CoNLL-U sent_id (e.g. train-s2)")
    group.add_argument("--text",    type=str,   help="Partial Hindi text to match")

    parser.add_argument("--max_show", type=int, default=99,
                        help="Maximum number of variants to include (default: all)")
    parser.add_argument("--no_browser", action="store_true",
                        help="Skip opening the browser")
    args = parser.parse_args()

    # ── Load treebank ──────────────────────────────────────────
    print("Loading treebank...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)

    print("Filtering valid sentences...")
    valid = [
        (s, sid, ctx)
        for s, sid, ctx in zip(sentences, sent_ids, contexts)
        if is_valid_treebank_sentence(s)
    ]
    print(f"  Valid sentences: {len(valid)}")

    # ── Build corpus-wide transitions ──────────────────────────
    print("Building corpus transitions...")
    all_valid_sents = [x[0] for x in valid]
    transitions = build_corpus_transitions(all_valid_sents)

    # ── Find the target sentence ───────────────────────────────
    target = None

    if args.index is not None:
        if args.index >= len(valid):
            print(f"ERROR: index {args.index} out of range (max {len(valid)-1})")
            return
        target = valid[args.index]

    elif args.sent_id is not None:
        for s, sid, ctx in valid:
            if sid == args.sent_id:
                target = (s, sid, ctx)
                break
        if target is None:
            print(f"ERROR: sent_id '{args.sent_id}' not found among valid sentences.")
            print("First 10 valid sent_ids:")
            for _, sid, _ in valid[:10]:
                print(f"  {sid}")
            return

    elif args.text is not None:
        query = args.text.strip()
        for s, sid, ctx in valid:
            surface = " ".join(t["word"] for t in s)
            if query in surface:
                target = (s, sid, ctx)
                break
        if target is None:
            print(f"ERROR: no valid sentence contains '{args.text}'")
            return

    sentence, sent_id, context = target
    construction_type = get_construction_type(sentence)

    # ── Generate variants ──────────────────────────────────────
    print(f"Generating variants for: {sent_id}")
    variants = generate_variants_subtrees(sentence, transitions)

    if not variants:
        print("No variants generated for this sentence.")
        return

    # Limit to max_show
    variants = variants[:args.max_show]
    print(f"  Reference construction : {construction_type}")
    print(f"  Variants generated     : {len(variants)}")

    # ── Print breakdown to terminal ────────────────────────────
    print_variant_breakdown(sentence, variants, sent_id, construction_type)

    # ── Build CoNLL-U output ───────────────────────────────────
    conllu_text = generate_conllu_for_sentence(
        sentence, variants, sent_id, construction_type
    )

    # ── Save CoNLL-U file ──────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_id   = (sent_id or "sentence").replace(":", "-").replace("/", "-")
    out_path  = os.path.join(OUTPUT_DIR, f"{safe_id}.conllu")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(conllu_text)

    print(f"CoNLL-U saved to : {out_path}")
    print(f"Lines            : {conllu_text.count(chr(10))}")

    # ── Instructions for viewer ────────────────────────────────
    print(f"\n{'='*65}")
    print(f"To view the dependency trees:")
    print(f"{'='*65}")
    print(f"  1. Open: {VIEWER_URL}")
    print(f"  2. Upload file: {os.path.abspath(out_path)}")
    print(f"     OR paste the contents of the file into the text box.")
    print(f"\n  The viewer will show:")
    print(f"  - Sentence 1  : Reference ({construction_type})")
    for i in range(1, min(len(variants)+1, 6)):
        print(f"  - Sentence {i+1:<2} : Variant {i:02d}")
    if len(variants) > 5:
        print(f"  - ... and {len(variants)-5} more variants")
    print()

    # ── Open browser ───────────────────────────────────────────
    if not args.no_browser:
        print(f"Opening viewer in browser...")
        webbrowser.open(VIEWER_URL)


if __name__ == "__main__":
    main()