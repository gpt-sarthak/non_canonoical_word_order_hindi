"""
demo_server.py

Flask API server for the Hindi sentence explorer demo.

Accepts a Hindi sentence, parses it with Stanza (UD Hindi model),
generates word-order variants, and scores each with:
  - Dependency Length (DL)
  - Trigram surprisal
  - LSTM surprisal
  - PCFG surprisal

Run:
    python demo_server.py

Then open:
    http://localhost:5001

First run will download the Stanza Hindi model (~500 MB) and build
the PCFG from the HUTB treebank (~30s, then cached).
"""

import os
import sys
import pickle
import math

import torch
import stanza
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Project root on path ──────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.hutb_loader import (
    generate_variants_subtrees,
    get_construction_type,
    is_valid_treebank_sentence,
)
from feature_extraction.trigram_features import (
    load_trigram_model,
    sentence_trigram_surprisal,
)
from models.lstm.model import LSTMLanguageModel
from feature_extraction.lstm_features import (
    load_vocab,
    load_lstm_model,
    sentence_lstm_surprisal,
    get_device,
)
from feature_extraction.pcfg_features import (
    build_pcfg_from_trees,
    extract_trees_from_conllu,
    sentence_log_prob_inside,
)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
TRIGRAM_PATH   = os.path.join(ROOT, "models", "trigram", "trigram.pkl")
LSTM_PATH      = os.path.join(ROOT, "models", "lstm", "base_model.pt")
VOCAB_PATH     = os.path.join(ROOT, "data", "processed", "vocab.pkl")
TREEBANK_PATH  = os.path.join(ROOT, "data", "raw", "UD_Hindi-HDTB", "hi_hdtb-ud-train.conllu")

# ─────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────
print("Loading trigram model...")
trigram_model = load_trigram_model(TRIGRAM_PATH)
print("  Done.")

print("Loading LSTM model...")
device = get_device()
raw_vocab = load_vocab(VOCAB_PATH)
word2idx  = raw_vocab["word2idx"]
lstm_model, device = load_lstm_model(LSTM_PATH, len(word2idx), device)
print("  Done.")

print("Building PCFG from HUTB treebank (~30s)...")
_trees = extract_trees_from_conllu(TREEBANK_PATH)
pcfg_model = build_pcfg_from_trees(_trees)
del _trees
print("  Done.")

print("Loading Stanza Hindi parser (downloads on first use)...")
stanza.download("hi", verbose=False)
nlp = stanza.Pipeline(
    "hi",
    processors="tokenize,pos,lemma,depparse",
    verbose=False,
    use_gpu=torch.cuda.is_available(),
)
print("  Done.")

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def stanza_to_tokens(doc):
    """Convert a Stanza document to our token-dict format."""
    tokens = []
    for sent in doc.sentences:
        for word in sent.words:
            tokens.append({
                "id":       word.id,
                "word":     word.text,
                "lemma":    word.lemma or word.text,
                "upos":     word.upos  or "_",
                "feats":    word.feats or "_",
                "head":     word.head,
                "deprel":   word.deprel or "_",
                "chunk_id": None,
            })
    return tokens


def compute_dl(tokens):
    """Dependency length = sum of |id - head| for non-root tokens."""
    return sum(abs(t["id"] - t["head"]) for t in tokens if t["head"] != 0)


def remap_dl(tokens, order):
    """Compute DL for a permuted variant given token order list."""
    tokens_by_id  = {t["id"]: t for t in tokens}
    position_map  = {orig_id: new_pos + 1 for new_pos, orig_id in enumerate(order)}
    total = 0
    for orig_id in order:
        orig_head = tokens_by_id[orig_id]["head"]
        if orig_head == 0:
            continue
        new_pos  = position_map[orig_id]
        new_head = position_map.get(orig_head, 0)
        total   += abs(new_pos - new_head)
    return total


def get_variant_construction(tokens, order):
    """Determine construction type (SOV/DOSV/IOSV) of a variant."""
    tokens_by_id = {t["id"]: t for t in tokens}
    position_map = {orig_id: new_pos + 1 for new_pos, orig_id in enumerate(order)}
    remapped = []
    for new_pos, orig_id in enumerate(order):
        tok = dict(tokens_by_id[orig_id])
        tok["id"]   = new_pos + 1
        orig_head   = tokens_by_id[orig_id]["head"]
        tok["head"] = position_map.get(orig_head, 0) if orig_head != 0 else 0
        remapped.append(tok)
    return get_construction_type(remapped)


def compute_arc_changes(tokens, order):
    """Return list of arcs whose length changed in the variant."""
    tokens_by_id = {t["id"]: t for t in tokens}
    position_map = {orig_id: new_pos + 1 for new_pos, orig_id in enumerate(order)}
    changes = []
    for t in tokens:
        if t["head"] == 0:
            continue
        ref_len = abs(t["id"] - t["head"])
        var_len = abs(position_map[t["id"]] - position_map.get(t["head"], 0))
        if ref_len != var_len:
            changes.append({
                "word":    t["word"],
                "head":    tokens_by_id[t["head"]]["word"],
                "deprel":  t["deprel"],
                "ref_len": ref_len,
                "var_len": var_len,
                "delta":   var_len - ref_len,
            })
    return changes


def score_pcfg(sentence_text):
    """Score a sentence with the PCFG model (-log P). Returns None on failure."""
    try:
        return round(sentence_log_prob_inside(sentence_text, pcfg_model), 2)
    except Exception:
        return None


def _analyze_sentence(sentence_text):
    """
    Core analysis logic used by both /analyze and /batch.
    Returns (result_dict, http_status_code).
    result_dict always has 'error' key if something went wrong.
    """
    sentence_text = sentence_text.strip()
    if not sentence_text:
        return {"error": "Please provide a Hindi sentence."}, 400

    # ── Parse with Stanza ──────────────────────────────────────
    try:
        doc = nlp(sentence_text)
    except Exception as e:
        return {"error": f"Parsing failed: {e}"}, 500

    tokens = stanza_to_tokens(doc)
    if not tokens:
        return {"error": "Could not tokenise the sentence."}, 400

    # ── Structural validation ──────────────────────────────────
    if not is_valid_treebank_sentence(tokens):
        root = next((t for t in tokens if t["head"] == 0), None)
        reasons = []
        if root and root["upos"] not in {"VERB", "AUX"}:
            reasons.append("the main verb must be a VERB or AUX")
        from data.hutb_loader import (
            has_negative_marker, is_declarative, is_projective,
            SUBJECT_RELS, OBJECT_RELS
        )
        if has_negative_marker(tokens):
            reasons.append("sentence contains a negative marker (नहीं/न/मत)")
        if not is_declarative(tokens):
            reasons.append("sentence appears to be a question")
        if not is_projective(tokens):
            reasons.append("dependency tree is non-projective")
        if root:
            root_deps = [t for t in tokens if t["head"] == root["id"]]
            if not any(t["deprel"] in SUBJECT_RELS for t in root_deps):
                reasons.append("no subject directly attached to the main verb")
            if not any(t["deprel"] in OBJECT_RELS for t in root_deps):
                reasons.append("no object directly attached to the main verb — try a transitive sentence")
            preverbal = [t for t in root_deps if t["id"] < root["id"] and t["deprel"] != "punct"]
            if len(preverbal) < 2:
                reasons.append("fewer than 2 preverbal phrases — need at least subject + object before the verb")

        msg = "Sentence doesn't meet requirements for variant generation"
        if reasons:
            msg += ": " + "; ".join(reasons) + "."
        else:
            msg += "."
        return {
            "error":  msg,
            "parsed": [{"word": t["word"], "upos": t["upos"], "deprel": t["deprel"], "head": t["head"]} for t in tokens],
        }, 422

    # ── Generate variants ──────────────────────────────────────
    variants_raw = generate_variants_subtrees(tokens, max_variants=20)
    if not variants_raw:
        return {"error": "No variants could be generated for this sentence."}, 422

    # ── Score reference ────────────────────────────────────────
    ref_surface      = " ".join(t["word"] for t in tokens)
    ref_trigram      = sentence_trigram_surprisal(ref_surface, trigram_model)
    ref_lstm         = sentence_lstm_surprisal(ref_surface, lstm_model, word2idx, device)
    ref_dl           = compute_dl(tokens)
    ref_pcfg         = score_pcfg(ref_surface)
    ref_construction = get_construction_type(tokens)

    # Arc list for SVG diagram
    arcs = [
        {
            "id":     t["id"],
            "word":   t["word"],
            "upos":   t["upos"],
            "head":   t["head"],
            "deprel": t["deprel"],
        }
        for t in tokens
    ]

    # ── Score variants ─────────────────────────────────────────
    ref_order = [t["id"] for t in tokens]   # sequential: [1,2,3,...]
    variants_out = []
    for v in variants_raw:
        tri   = sentence_trigram_surprisal(v["sentence"], trigram_model)
        lst   = sentence_lstm_surprisal(v["sentence"], lstm_model, word2idx, device)
        dl_v  = remap_dl(tokens, v["order"])
        pcfg_v = score_pcfg(v["sentence"])
        ctype  = get_variant_construction(tokens, v["order"])
        arc_ch = compute_arc_changes(tokens, v["order"])

        # Which words moved (position in variant ≠ original id)
        moved_ids = {
            tok_id for i, tok_id in enumerate(v["order"])
            if tok_id != ref_order[i]
        }

        variants_out.append({
            "sentence":     v["sentence"],
            "trigram":      round(tri,   2),
            "lstm":         round(lst,   2),
            "dl":           dl_v,
            "pcfg":         pcfg_v,
            "construction": ctype,
            "order":        v["order"],
            "moved_ids":    list(moved_ids),
            "arc_changes":  arc_ch,
        })

    # ── Parse info for display ────────────────────────────────
    parse_info = [
        {
            "word":   t["word"],
            "upos":   t["upos"],
            "deprel": t["deprel"],
            "head":   t["head"],
            "id":     t["id"],
        }
        for t in tokens
    ]

    return {
        "reference": {
            "sentence":     ref_surface,
            "trigram":      round(ref_trigram, 2),
            "lstm":         round(ref_lstm,    2),
            "dl":           ref_dl,
            "pcfg":         ref_pcfg,
            "construction": ref_construction,
            "arcs":         arcs,
        },
        "variants": variants_out,
        "parse":    parse_info,
    }, 200


# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
CORS(app)

REPORTS_DIR = os.path.join(ROOT, "reports")


@app.route("/")
def index():
    return send_from_directory(REPORTS_DIR, "sentence_explorer.html")


@app.route("/pipeline_analysis.html")
def pipeline():
    return send_from_directory(REPORTS_DIR, "pipeline_analysis.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    sentence_text = (data.get("sentence") or "").strip()
    result, status = _analyze_sentence(sentence_text)
    return jsonify(result), status


@app.route("/batch", methods=["POST"])
def batch():
    """
    Analyze multiple sentences at once.
    Input:  {"sentences": ["sent1", "sent2", ...]}
    Output: list of compact result dicts (one per sentence)
    """
    data = request.get_json(force=True)
    sentences = data.get("sentences") or []
    if not isinstance(sentences, list):
        return jsonify({"error": "sentences must be a list"}), 400

    out = []
    for sent_text in sentences:
        sent_text = (sent_text or "").strip()
        if not sent_text:
            out.append({"sentence": sent_text, "error": "empty"})
            continue
        result, status = _analyze_sentence(sent_text)
        if status != 200:
            out.append({"sentence": sent_text, "error": result.get("error", "unknown error")})
            continue
        ref = result["reference"]
        vs  = result["variants"]
        row = {
            "sentence":      ref["sentence"],
            "construction":  ref["construction"],
            "n_variants":    len(vs),
            "ref_trigram":   ref["trigram"],
            "ref_lstm":      ref["lstm"],
            "ref_dl":        ref["dl"],
            "ref_pcfg":      ref["pcfg"],
            "min_trigram":   min((v["trigram"] for v in vs), default=None),
            "min_lstm":      min((v["lstm"]    for v in vs), default=None),
            "min_dl":        min((v["dl"]      for v in vs), default=None),
            "min_pcfg":      min((v["pcfg"]    for v in vs if v["pcfg"] is not None), default=None),
            "error":         None,
        }
        out.append(row)
    return jsonify(out), 200


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Hindi Sentence Explorer")
    print("  Open: http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
