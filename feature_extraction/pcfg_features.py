"""
pcfg_features.py

PCFG surprisal for the Ranjan & van Schijndel (2024) replication.

Design
------
The paper uses the Modelblocks incremental constituency parser to compute
per-word PCFG surprisal, then sums them to get sentence-level PCFG surprisal.
By the telescoping identity:

    Σ_k S(w_k)  =  -log P(w_1 … w_n)

the sum of per-word incremental surprisals equals the negative log-probability
of the full sentence.  We therefore compute -log P(sentence) directly using the
Inside algorithm, which gives an identical result.

Grammar
-------
We induce a 2-level PCFG from the HDTB chunk (shallow-parse) annotations
already present in the CoNLL-U MISC field (ChunkId / ChunkType), following
Yadav, Vaidya & Husain (2017):

    S   → C_1  C_2  …  C_k          sentence-level rule (chunk sequence)
    C_i → w_1  w_2  …  w_m          chunk-internal rule (word sequence)

where C_i is the base chunk label (NP, VGF, JJP, …).

Fast scoring
------------
Because our dataset items carry the original CoNLL-U token list (with
chunk_id per token) AND the variant token-ID ordering, we can reconstruct
the exact chunk sequence for both reference and variant sentences directly —
no search needed.  Scoring is O(k) per sentence (k = number of chunks ≈ 3-8).

The general Inside-algorithm fallback (O(n²·beam)) is kept for sentences
whose chunk structure is unavailable (e.g. externally generated strings).

5-fold cross-validation
-----------------------
Following the paper: all ~13,000 HUTB train trees are split into 5 folds.
For each fold, the PCFG is trained on the other four folds and used to score
the held-out sentences.  This ensures no reference sentence is scored by a
model that trained on it.  Variants are scored with the full PCFG (they are
not in the HUTB, so there is no leakage).
"""

import re
import math
import random
from collections import defaultdict


# ── 1. TREE EXTRACTION FROM CoNLL-U ─────────────────────────────────────────

def _chunk_base(chunk_id):
    """'NP2' → 'NP',  'VGF' → 'VGF',  None → None."""
    if chunk_id is None:
        return None
    return re.sub(r"\d+$", "", chunk_id)


def tokens_to_chunks(tokens):
    """
    Convert a list of token dicts (with 'chunk_id' and 'word' keys) into
    an ordered list of (base_label, [word, …]) tuples — one per chunk.

    Tokens without a chunk_id are assigned to a synthetic 'UNK_CHUNK'.
    Chunk order follows the token order in the list.
    """
    chunks = []
    seen_ids = []
    id_to_words = defaultdict(list)
    id_to_label = {}

    for tok in tokens:
        cid   = tok.get("chunk_id")
        label = _chunk_base(cid) or "UNK_CHUNK"
        word  = tok["word"].lower()

        if cid not in id_to_label:
            id_to_label[cid] = label
            seen_ids.append(cid)
        id_to_words[cid].append(word)

    for cid in seen_ids:
        chunks.append((id_to_label[cid], id_to_words[cid]))

    return chunks


def extract_trees_from_conllu(filepath):
    """
    Read a CoNLL-U file and return a list of constituency tree dicts.
    Each dict:
        sentence   : surface string
        s_rule     : ("S", [label, label, …])
        chunk_rules: [(label, [word, …]), …]
    """
    trees = []
    current_tokens = []

    def flush(toks):
        if not toks:
            return
        chunks = tokens_to_chunks(toks)
        if not chunks:
            return
        s_children  = [lbl for lbl, _ in chunks]
        surface     = " ".join(t["word"] for t in toks)
        trees.append({
            "sentence":    surface,
            "s_rule":      ("S", s_children),
            "chunk_rules": chunks,
        })

    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                flush(current_tokens)
                current_tokens = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 10:
                continue
            if "-" in parts[0] or "." in parts[0]:
                continue

            chunk_id = None
            for kv in parts[9].split("|"):
                if kv.startswith("ChunkId="):
                    chunk_id = kv.split("=", 1)[1]
                    break

            current_tokens.append({"word": parts[1], "chunk_id": chunk_id})

    flush(current_tokens)
    return trees


# ── 2. PCFG INDUCTION ───────────────────────────────────────────────────────

class PCFG:
    """
    Probabilistic Context-Free Grammar with add-1 (Laplace) smoothing.

    rule_counts[lhs][rhs_tuple] = int
    lhs_counts[lhs]             = int   (total observations for lhs)
    log_probs[lhs][rhs_tuple]   = float (log probability, computed after training)
    """

    def __init__(self):
        self.rule_counts = defaultdict(lambda: defaultdict(int))
        self.lhs_counts  = defaultdict(int)

    def add_tree(self, tree):
        lhs, rhs = tree["s_rule"]
        self.rule_counts[lhs][tuple(rhs)] += 1
        self.lhs_counts[lhs] += 1

        for lhs, words in tree["chunk_rules"]:
            self.rule_counts[lhs][tuple(words)] += 1
            self.lhs_counts[lhs] += 1

    def compute_log_probs(self):
        self.log_probs = {}
        for lhs, rhs_dict in self.rule_counts.items():
            total   = self.lhs_counts[lhs]
            n_rules = len(rhs_dict)
            self.log_probs[lhs] = {}
            for rhs_t, cnt in rhs_dict.items():
                self.log_probs[lhs][rhs_t] = math.log(
                    (cnt + 1) / (total + n_rules)
                )
        # Store floor denominator per lhs for unseen rules
        self._floor = {}
        for lhs in self.log_probs:
            total   = self.lhs_counts[lhs]
            n_rules = len(self.log_probs[lhs])
            self._floor[lhs] = math.log(1.0 / (total + n_rules + 1))

    def log_prob(self, lhs, rhs_tuple):
        """log P(rhs | lhs), with add-1 floor for unseen rules."""
        if lhs not in self.log_probs:
            return -30.0
        return self.log_probs[lhs].get(rhs_tuple, self._floor.get(lhs, -30.0))


# ── 3. SCORING FUNCTIONS ─────────────────────────────────────────────────────

def score_chunks(chunks, pcfg):
    """
    Fast O(k) scoring given a known chunk sequence.

    log P(sentence) = log P(S → label_1 … label_k)
                    + Σ_i log P(label_i → word_1 … word_m)

    Returns -log P(sentence) as the PCFG surprisal.
    """
    labels     = tuple(lbl for lbl, _ in chunks)
    log_p      = pcfg.log_prob("S", labels)
    for lbl, words in chunks:
        log_p += pcfg.log_prob(lbl, tuple(words))
    return -log_p   # surprisal = -log P


def score_from_tokens(tokens, pcfg):
    """Score a sentence given its token list (tokens have 'chunk_id')."""
    chunks = tokens_to_chunks(tokens)
    return score_chunks(chunks, pcfg)


def score_variant_from_tokens_and_order(tokens, order, pcfg):
    """
    Score a variant sentence.

    tokens : original token list (with chunk_id, word, id)
    order  : list of token IDs in the variant's surface order
    """
    tok_by_id   = {t["id"]: t for t in tokens}
    reordered   = [tok_by_id[tid] for tid in order if tid in tok_by_id]
    chunks      = tokens_to_chunks(reordered)
    return score_chunks(chunks, pcfg)


# ── 4. INSIDE ALGORITHM (fallback for unknown chunk structure) ───────────────

def sentence_log_prob_inside(sentence_str, pcfg, beam=200):
    """
    Compute -log P(sentence) via beam Inside DP.

    Used as a fallback when token/chunk structure is unavailable
    (e.g. for external sentences not in the HUTB).

    Because Σ_k S(w_k) = -log P(sentence) (telescoping identity),
    this gives the same result as summing per-word Earley surprisals.
    """
    words = sentence_str.lower().split()
    n     = len(words)
    if n == 0:
        return 0.0

    NEG_INF      = float("-inf")
    chunk_labels = [lhs for lhs in pcfg.log_probs if lhs != "S"]

    # Precompute chunk log-prob for every (label, span)
    chunk_logp = [[{} for _ in range(n + 1)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n + 1):
            span = tuple(words[i:j])
            for lbl in chunk_labels:
                chunk_logp[i][j][lbl] = pcfg.log_prob(lbl, span)

    # Beam DP: pos_states[i] = list of (cumulative_log_prob, label_seq_tuple)
    pos_states    = [None] * (n + 1)
    pos_states[0] = [(0.0, ())]

    for i in range(n):
        if not pos_states[i]:
            continue
        for j in range(i + 1, n + 1):
            for lbl in chunk_labels:
                chunk_lp = chunk_logp[i][j].get(lbl, NEG_INF)
                if chunk_lp == NEG_INF:
                    continue
                for prev_lp, prev_seq in pos_states[i]:
                    new_entry = (prev_lp + chunk_lp, prev_seq + (lbl,))
                    if pos_states[j] is None:
                        pos_states[j] = []
                    pos_states[j].append(new_entry)

        # Beam pruning
        for k in range(i + 1, n + 1):
            if pos_states[k] and len(pos_states[k]) > beam:
                pos_states[k].sort(key=lambda x: x[0], reverse=True)
                pos_states[k] = pos_states[k][:beam]

    complete = pos_states[n] or []
    if not complete:
        return float("inf")   # unparseable → infinite surprisal

    log_probs_full = [
        chunk_sum_lp + pcfg.log_prob("S", label_seq)
        for chunk_sum_lp, label_seq in complete
    ]
    return -_log_sum_exp(log_probs_full)


def _log_sum_exp(log_probs):
    if not log_probs:
        return float("-inf")
    m = max(log_probs)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(lp - m) for lp in log_probs))


# ── 5. PCFG TRAINING ─────────────────────────────────────────────────────────

def build_pcfg_from_trees(trees):
    pcfg = PCFG()
    for tree in trees:
        pcfg.add_tree(tree)
    pcfg.compute_log_probs()
    return pcfg


# ── 6. 5-FOLD CROSS-VALIDATION ───────────────────────────────────────────────

def compute_pcfg_surprisal_cv(all_hutb_trees, study_sentences, n_folds=5, seed=42):
    """
    5-fold CV following the paper.

    Critical design: BOTH the reference sentence AND all its variants must be
    scored using the SAME fold PCFG (the one that did not train on that reference).
    Scoring references with the fold model but variants with the full model
    creates a systematic bias (fold model is weaker → references get artificially
    higher surprisal → spurious accuracy).

    Args:
        all_hutb_trees  : all ~13,000 tree dicts from the HUTB train file
        study_sentences : set/list of reference surface strings
        n_folds         : 5 (paper default)
        seed            : reproducibility

    Returns:
        sent_to_pcfg  : {surface_string → (PCFG surprisal, fold_pcfg)}
                        fold_pcfg is the PCFG that should also score this
                        sentence's variants (ensures fair comparison)
        full_pcfg     : PCFG trained on ALL trees (for sentences not in HUTB)
    """
    rng = random.Random(seed)

    surface_to_tree = {t["sentence"]: t for t in all_hutb_trees}

    study_trees = [surface_to_tree[s] for s in study_sentences if s in surface_to_tree]
    missing     = [s for s in study_sentences if s not in surface_to_tree]

    print(f"  Study sentences matched to HUTB trees : {len(study_trees)}")
    print(f"  Study sentences NOT in HUTB trees     : {len(missing)}")

    # Shuffle and fold
    rng.shuffle(study_trees)
    fold_size = max(1, len(study_trees) // n_folds)
    folds = [
        study_trees[k * fold_size : (k + 1) * fold_size if k < n_folds - 1 else len(study_trees)]
        for k in range(n_folds)
    ]

    # Full PCFG (for sentences not in HUTB — no leakage risk)
    print("  Training full PCFG on all HUTB trees ...")
    full_pcfg = build_pcfg_from_trees(all_hutb_trees)

    # sent_to_pcfg: surface → (surprisal, pcfg_to_use_for_variants)
    sent_to_pcfg = {}

    # Sentences not in HUTB: use full PCFG for both ref and variants
    for sent in missing:
        surp = score_chunks(
            tokens_to_chunks([{"word": w, "chunk_id": None} for w in sent.split()]),
            full_pcfg
        )
        sent_to_pcfg[sent] = (surp, full_pcfg)

    # CV folds
    print(f"  Running {n_folds}-fold CV on {len(study_trees)} study trees ...")
    for k, test_fold in enumerate(folds):
        test_surfaces = {t["sentence"] for t in test_fold}
        train_trees   = [t for t in all_hutb_trees if t["sentence"] not in test_surfaces]
        fold_pcfg     = build_pcfg_from_trees(train_trees)
        print(f"    Fold {k+1}/{n_folds}: train={len(train_trees)}, test={len(test_fold)}")

        for tree in test_fold:
            surp = score_chunks(tree["chunk_rules"], fold_pcfg)
            # Store both the surprisal AND the fold_pcfg so variants can be
            # scored with the SAME model
            sent_to_pcfg[tree["sentence"]] = (surp, fold_pcfg)

    return sent_to_pcfg, full_pcfg


# ── 7. PUBLIC API ─────────────────────────────────────────────────────────────

def compute_pcfg_features(dataset, all_hutb_trees, n_folds=5):
    """
    Compute PCFG surprisal for all reference-variant pairs.

    Args:
        dataset        : list of pair dicts from hutb_loader
                         (each has 'reference', 'variant', 'tokens', 'order')
        all_hutb_trees : tree dicts from extract_trees_from_conllu() on train file
        n_folds        : CV folds (default 5)

    Returns:
        list of pair dicts with added keys:
            pcfg_reference, pcfg_variant, delta_pcfg
            delta_pcfg = pcfg_reference - pcfg_variant
            (negative = reference more probable, matches paper's delta convention
             where Reference - Variant < 0 means Reference is better)
    """
    study_sentences = list({d["reference"] for d in dataset})

    print(f"[PCFG] {len(study_sentences)} unique reference sentences")
    print(f"[PCFG] {len(dataset)} total pairs")

    # 5-fold CV for reference sentences
    ref_surprisal, full_pcfg = compute_pcfg_surprisal_cv(
        all_hutb_trees, study_sentences, n_folds=n_folds
    )

    # Score variants using fast direct chunk scoring
    # Each dataset item carries 'tokens' (with chunk_id) and 'order'
    # CRITICAL: use the SAME fold_pcfg that scored the reference to avoid CV bias
    print("[PCFG] Scoring variants (fast direct chunk scoring) ...")

    results = []
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"  {i}/{len(dataset)}")

        ref_entry = ref_surprisal.get(item["reference"], (float("nan"), full_pcfg))
        ref_s, fold_pcfg = ref_entry  # unpack surprisal and the fold's PCFG

        # Fast variant scoring: reconstruct chunk sequence from token order
        # Use fold_pcfg (same model that scored this reference) to avoid bias
        tokens = item.get("tokens", [])
        order  = item.get("order",  [])

        if tokens and order:
            var_s = score_variant_from_tokens_and_order(tokens, order, fold_pcfg)
        else:
            # Fallback: use Inside algorithm on the surface string
            var_s = sentence_log_prob_inside(item["variant"], fold_pcfg)

        results.append({
            **item,
            "pcfg_reference": ref_s,
            "pcfg_variant":   var_s,
            "delta_pcfg":     ref_s - var_s,
        })

    return results


# ── 8. SANITY CHECK ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    TRAIN_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"

    print("=== PCFG Sanity Check ===\n")

    print(f"Extracting trees from {TRAIN_PATH} ...")
    trees = extract_trees_from_conllu(TRAIN_PATH)
    print(f"  {len(trees)} trees extracted\n")

    print("Example trees:")
    for t in trees[:3]:
        print(f"  {t['sentence'][:55]}")
        print(f"  S → {' '.join(t['s_rule'][1])}")
        chunks_summary = [(lbl, len(words)) for lbl, words in t['chunk_rules']]
        print(f"  Chunks: {chunks_summary}\n")

    print("Training PCFG ...")
    pcfg = build_pcfg_from_trees(trees)
    total_rules = sum(len(v) for v in pcfg.log_probs.values())
    print(f"  Non-terminals: {len(pcfg.log_probs)},  Rules: {total_rules}\n")

    print("Benchmarking fast chunk scoring (100 sentences) ...")
    test_trees = trees[:100]
    t0 = time.time()
    for tree in test_trees:
        score_chunks(tree["chunk_rules"], pcfg)
    elapsed = time.time() - t0
    print(f"  {elapsed*1000:.1f} ms total  ({elapsed/100*1000:.3f} ms/sentence)")
    print(f"  Estimate for 92,000 sentences: {92000 * elapsed/100 / 60:.1f} minutes\n")

    print("Sample surprisal values:")
    for tree in trees[:5]:
        s = score_chunks(tree["chunk_rules"], pcfg)
        print(f"  surprisal={s:.3f}  |  {tree['sentence'][:55]}")

    print("\nRunning 5-fold CV on first 100 study sentences ...")
    study = [t["sentence"] for t in trees[:100]]
    t0 = time.time()
    ref_s, _ = compute_pcfg_surprisal_cv(trees, study, n_folds=5)
    print(f"  CV completed in {time.time()-t0:.1f}s")
    print("\nSample CV surprisal values:")
    for sent in study[:5]:
        surp, _ = ref_s[sent]
        print(f"  CV surprisal={surp:.3f}  |  {sent[:50]}")

    print("\n=== Done ===")
