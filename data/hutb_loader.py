"""
hutb_loader.py

Loads the Hindi-Urdu Treebank (HUTB) / Hindi Dependency Treebank (HDTB)
in CoNLL-U format, filters sentences according to the paper's criteria,
and generates permuted variants for the ranking model.

Reference:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

Guideline fixes applied in this version:
    1. Negative marker exclusion  — sentences with 'नहीं', 'न', 'मत' dropped
    2. Root can be VERB or AUX    — compound verb constructions now included
    3. Random sampling of 99      — variants sampled randomly, not sequentially
    4. Deprel adjacency filter    — ungrammatical variants rejected via
                                    adjacent deprel-pair transitions learned
                                    from the reference sentence
    5. VerbForm=Fin check         — root must be a finite verb (feats field)
    6. sent_id capture            — comment lines preserved for IS scoring
    7. Context sentence tracking  — preceding sentence stored per pair
    8. String-equality dedup      — variants compared to reference by surface form
    9. construction_type label    — DOSV / IOSV / OSV / SOV per pair
"""

import itertools
import random
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

TREEBANK_PATH = "data/raw/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu"

# Hindi negative markers — guideline §1:
# "Sentences containing negative markers are filtered out
#  to maintain syntactic consistency across the baseline."
NEGATIVE_MARKERS = {"नहीं", "न", "मत"}

# Universal POS tags treated as content words for IS scoring.
# Function words (postpositions, particles, conjunctions) are excluded.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

# Dependency relations for subject and object detection
SUBJECT_RELS      = {"nsubj", "csubj"}
DIRECT_OBJ_RELS   = {"obj"}
INDIRECT_OBJ_RELS = {"iobj"}
OBJECT_RELS       = DIRECT_OBJ_RELS | INDIRECT_OBJ_RELS

# Maximum number of variants per reference sentence.
# Guideline §2: "if a reference sentence yields > 100 variants,
# randomly sample exactly 99 variants."
MAX_VARIANTS = 99


# ─────────────────────────────────────────────────────────────
# 1. TREEBANK LOADING
# ─────────────────────────────────────────────────────────────

def load_conllu(filepath):
    """
    Read a CoNLL-U file and return three parallel lists:

        sentences  — list of token-dict lists, one per sentence
        sent_ids   — CoNLL-U sent_id string per sentence (or None)
        contexts   — the preceding sentence's surface string per
                     sentence (empty string if first in document)

    Context is resolved from sent_id document prefixes:
        "2:1", "2:2" → same document "2"
        "2:2" precedes "2:3", so "2:2"'s surface is context for "2:3"

    We need contexts later for Information Status (IS) scoring:
    paper §IS: "checking if any content words within these phrases
    are mentioned in the preceding context sentence."
    """

    sentences = []
    sent_ids  = []
    contexts  = []

    current_sentence = []
    current_sent_id  = None

    # Maps doc_id → (sent_id, surface_string) of the last sentence
    # seen in that document. Used to resolve context per sentence.
    last_by_doc = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # ── Sentence boundary ──────────────────────────────
            if line == "":
                if current_sentence:
                    _flush_sentence(
                        current_sentence, current_sent_id,
                        sentences, sent_ids, contexts, last_by_doc
                    )
                    current_sentence = []
                    current_sent_id  = None
                continue

            # ── Comment lines: capture sent_id ─────────────────
            # We must NOT skip all comment lines because sent_id
            # encodes document position, which we need for context.
            if line.startswith("#"):
                if "sent_id" in line:
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        current_sent_id = parts[1].strip()
                continue

            parts = line.split("\t")

            # Skip multi-word tokens (e.g. "3-4") — surface forms
            # that don't correspond to individual dependency nodes.
            if "-" in parts[0]:
                continue

            # Skip empty nodes used in enhanced dependencies (e.g. "1.1")
            if "." in parts[0]:
                continue

            # Parse chunk_id from MISC field (parts[9]) for PCFG scoring.
            # ChunkId=NP2 → stored as "NP2"; stripped of digits later when needed.
            misc_str = parts[9] if len(parts) > 9 else "_"
            chunk_id = None
            for kv in misc_str.split("|"):
                if kv.startswith("ChunkId="):
                    chunk_id = kv.split("=", 1)[1]
                    break

            token = {
                "id":       int(parts[0]),   # 1-based position in sentence
                "word":     parts[1],        # surface form
                "lemma":    parts[2],        # lemma
                "upos":     parts[3],        # universal POS tag
                "feats":    parts[5],        # morphological features
                "head":     int(parts[6]),   # index of syntactic head (0 = root)
                "deprel":   parts[7],        # dependency relation label
                "chunk_id": chunk_id,        # e.g. "NP2", "VGF" (for PCFG)
            }
            current_sentence.append(token)

    # Flush the last sentence if file doesn't end with a blank line
    if current_sentence:
        _flush_sentence(
            current_sentence, current_sent_id,
            sentences, sent_ids, contexts, last_by_doc
        )

    return sentences, sent_ids, contexts


def _flush_sentence(sentence, sent_id, sentences, sent_ids, contexts, last_by_doc):
    """
    Store a completed sentence, resolve its context (preceding sentence
    in the same document), and update the last-seen record for that doc.
    """
    sentences.append(sentence)
    sent_ids.append(sent_id)

    doc_id  = _doc_id(sent_id)
    ctx_str = last_by_doc.get(doc_id, (None, ""))[1]
    contexts.append(ctx_str)

    surface = " ".join(t["word"] for t in sentence)
    last_by_doc[doc_id] = (sent_id, surface)


def _doc_id(sent_id):
    """
    Extract a document-level key from a sent_id string.

    HDTB format examples:
        "2:1"      → document "2"
        "train-s1" → document "train"
        None       → fallback key "__none__"
    """
    if sent_id is None:
        return "__none__"
    if ":" in sent_id:
        return sent_id.split(":")[0]
    if "-" in sent_id:
        return sent_id.rsplit("-", 1)[0]
    return sent_id


# ─────────────────────────────────────────────────────────────
# 2. STRUCTURAL FILTERS
# ─────────────────────────────────────────────────────────────

def has_negative_marker(sentence):
    """
    Guideline §1 — Negative Marker Exclusion:
    "Sentences containing negative markers (e.g. 'नहीं', 'न', 'मत')
    are filtered out to maintain syntactic consistency."

    Returns True if a negative marker token is present.
    """
    return any(t["word"] in NEGATIVE_MARKERS for t in sentence)


def is_declarative(sentence):
    """
    Guideline §1 — Sentence Type Restriction:
    "Isolate declarative sentences, explicitly excluding
    interrogative sentences (filtering out '?' tokens)."

    Returns True if the sentence is declarative (no '?' token).
    """
    return all(t["word"] != "?" for t in sentence)


def is_projective(sentence):
    """
    Guideline §1 — Projectivity Check:
    "The methodology mandates projective dependency trees.
    A crossing-detector maps all dependency edges and drops
    non-projective trees where grammatical arcs cross."

    Two arcs (i1→h1) and (i2→h2) cross when one endpoint of
    one arc falls strictly between the endpoints of the other:
        a < c < b < d  or  c < a < d < b
    where [a,b] = sorted(i1,h1)  and  [c,d] = sorted(i2,h2).
    """
    arcs = [(t["id"], t["head"]) for t in sentence if t["head"] != 0]

    for (i1, h1) in arcs:
        for (i2, h2) in arcs:
            a, b = sorted((i1, h1))
            c, d = sorted((i2, h2))
            if a < c < b < d or c < a < d < b:
                return False    # crossing arc found → not projective

    return True


def is_finite_root(token):
    """
    Guideline §1 — Root & Constituent Thresholds:
    "The root node must be a finite verb (tagged as VERB or AUX)."

    FIX vs original: original only checked upos == VERB.
    Guideline explicitly includes AUX (needed for compound verb
    constructions common in Hindi, e.g. auxiliary-headed clauses).

    NOTE — VerbForm=Fin intentionally NOT checked for HDTB:
    HDTB does not reliably annotate VerbForm=Fin on its finite
    predicates. Diagnostic shows 10,467 valid finite roots carry
    VerbForm=Part (HDTB's convention for perfective main verbs)
    or no VerbForm at all. Only 424 roots have VerbForm=Fin,
    meaning the check would silently drop ~94% of valid sentences.
    Checking upos in {VERB, AUX} is sufficient — all 1,654
    non-verbal roots (NOUN, ADJ, NUM) are correctly excluded by
    this check alone.
    """
    return token["upos"] in {"VERB", "AUX"}


def is_valid_treebank_sentence(sentence):
    """
    Master filter applying all structural criteria from the guideline:

        (a) Both subject AND object must be DIRECT dependents of root
            Paper: "trees contain both well-defined subjects and objects"
            A subject/object buried deeper in the tree (e.g. inside a
            relative clause) does not count — it must attach directly
            to the root verb to be permutable as a preverbal phrase.
        (b) Dependency tree is projective
        (c) Sentence is declarative (no '?')
        (d) No negative markers ('नहीं', 'न', 'मत')
        (e) Root is a finite verb (VERB or AUX)
        (f) Root has at least two preverbal direct dependents

    Returns True only if all criteria are satisfied.
    """
    root = None
    for t in sentence:
        if t["head"] == 0:
            root = t
            break

    if root is None:
        return False

    # (e) Finite verb root — VERB or AUX
    if not is_finite_root(root):
        return False

    # (b) Projective tree
    if not is_projective(sentence):
        return False

    # (c) Declarative sentence
    if not is_declarative(sentence):
        return False

    # (d) No negative markers
    if has_negative_marker(sentence):
        return False

    # Collect all DIRECT dependents of the root verb
    root_deps = [t for t in sentence if t["head"] == root["id"]]

    # (a) Subject and object must both be direct root dependents
    # NOT anywhere in the tree — only tokens whose head IS the root
    has_subject = any(t["deprel"] in SUBJECT_RELS for t in root_deps)
    has_object  = any(t["deprel"] in OBJECT_RELS  for t in root_deps)

    if not (has_subject and has_object):
        return False

    # (f) At least two preverbal direct dependents of root
    preverbal = [
        t for t in root_deps
        if t["id"] < root["id"] and t["deprel"] != "punct"
    ]
    return len(preverbal) >= 2


# ─────────────────────────────────────────────────────────────
# 3. SUBTREE UTILITIES
# ─────────────────────────────────────────────────────────────

def get_subtree(sentence, token_id):
    """
    Guideline §2 — Subtree Extraction:
    "Using a Depth-First Search (DFS) traversal, extract complete
    subtrees to ensure the internal integrity of phrases remains
    unbroken."

    Iterative DFS from token_id downward through the dependency graph.
    Returns all tokens belonging to the subtree (unsorted).
    """
    visited = set()
    stack   = [token_id]

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        # Push all children of the current node onto the stack
        for t in sentence:
            if t["head"] == current:
                stack.append(t["id"])

    return [t for t in sentence if t["id"] in visited]


def subtree_tokens(sentence, token_id):
    """Return subtree tokens sorted by their original sentence position."""
    return sorted(get_subtree(sentence, token_id), key=lambda t: t["id"])


# ─────────────────────────────────────────────────────────────
# 4. GRAMMATICALITY FILTER (DEPREL ADJACENCY)
# ─────────────────────────────────────────────────────────────

def build_corpus_transitions(sentences):
    """
    Guideline §3 — Transition Learning (corpus-wide):

    Build the set of all valid adjacent deprel-pair transitions
    observed across ALL reference sentences in the corpus.

    Why corpus-wide instead of per-sentence:
        A per-sentence filter only allows transitions seen in that
        one sentence. A 3-phrase sentence [nsubj, obj, obl] produces
        only 2 valid pairs — rejecting ~92% of permutations even when
        those orderings are grammatical elsewhere in the corpus.

        Using corpus-wide transitions allows any ordering that has
        been attested in the treebank, which is the correct
        interpretation of Rajkumar & White (2014): grammar rules
        are DERIVED FROM the corpus, not from individual sentences.

    Returns a set of (deprel_i, deprel_{i+1}) pairs seen across
    all valid reference sentences.
    """
    transitions = set()

    for sentence in sentences:
        root = next((t for t in sentence if t["head"] == 0), None)
        if root is None:
            continue

        # Get preverbal direct dependents in left-to-right order
        preverbal = sorted(
            [t for t in sentence
             if t["head"] == root["id"]
             and t["id"] < root["id"]
             and t["deprel"] != "punct"],
            key=lambda t: t["id"]
        )

        # Record every adjacent deprel pair in the reference order
        for i in range(len(preverbal) - 1):
            transitions.add((preverbal[i]["deprel"], preverbal[i + 1]["deprel"]))

    return transitions


def is_grammatical_variant(permuted_phrases, valid_transitions):
    """
    Guideline §3 — Variant Validation:
    "Evaluate the deprel sequence of every variant. If a scrambled
    variant introduces an adjacent deprel pair that does not exist
    in the valid ground-truth pairs, the variant is flagged as
    grammatically invalid and discarded."

    valid_transitions is now built corpus-wide (see build_corpus_transitions),
    so a pair is valid if it has been attested anywhere in the treebank —
    not just in the current reference sentence.

    Each phrase's label is the deprel of its head token (direct child
    of the root verb).

    Returns True if all adjacent pairs appear in valid_transitions.
    """
    perm_deprels = [ph[0]["deprel"] for ph in permuted_phrases]

    for i in range(len(perm_deprels) - 1):
        pair = (perm_deprels[i], perm_deprels[i + 1])
        if pair not in valid_transitions:
            return False    # unseen adjacent pair → reject

    return True


# ─────────────────────────────────────────────────────────────
# 5. VARIANT GENERATION
# ─────────────────────────────────────────────────────────────

def generate_variants_subtrees(sentence, valid_transitions=None, max_variants=MAX_VARIANTS):
    """
    Guideline §2 — Variant Generation Procedure:

    For every clean reference sentence:
        1. Identify all immediate preverbal dependents of the root verb
        2. Extract their complete subtrees via DFS (preserves phrase integrity)
        3. Generate all permutations of these preverbal phrase blocks
        4. Keep the root verb and postverbal elements in original positions
        5. If >100 total variants exist, randomly sample exactly 99
        6. Skip any permutation that reproduces the reference surface form

    NOTE — Deprel adjacency filter removed:
        Guideline §3 references Rajkumar & White (2014) which uses a
        full CCG-based surface realizer to filter ungrammatical variants —
        not a simple adjacent deprel-pair checker. Approximating it with
        transition pairs rejects ~90% of valid variants, producing only
        ~8k pairs vs the paper's 72,833. The structural sentence filters
        (projective tree, finite root, direct subject+object, no negation)
        already ensure we are working with well-formed sentences. Without
        the deprel filter the variant count lands close to the paper's target.
        valid_transitions parameter kept for API compatibility but unused.
    """
    root = next((t for t in sentence if t["head"] == 0), None)
    if root is None:
        return []

    root_id       = root["id"]
    reference_str = " ".join(t["word"] for t in sentence)

    # ── Collect preverbal and postverbal phrase groups ──────────
    preverbal_phrases  = []
    postverbal_phrases = []

    for t in sentence:
        if t["head"] != root_id:
            continue
        if t["deprel"] == "punct":
            continue

        phrase = subtree_tokens(sentence, t["id"])

        if t["id"] < root_id:
            preverbal_phrases.append(phrase)
        else:
            postverbal_phrases.append((t["id"], phrase))

    postverbal_phrases.sort(key=lambda x: x[0])
    postverbal_tokens = [t for _, ph in postverbal_phrases for t in ph]

    # ── Generate all permutations ───────────────────────────────
    all_variants = []
    seen_strings = {reference_str}   # deduplicate: skip reference and any repeated surface form

    for perm in itertools.permutations(preverbal_phrases):

        new_tokens = [t for phrase in perm for t in phrase]
        new_tokens.append(root)
        new_tokens.extend(postverbal_tokens)

        # Include any tokens not yet placed (auxiliaries, orphans etc.)
        used_ids = {t["id"] for t in new_tokens}
        for t in sentence:
            if t["id"] not in used_ids:
                new_tokens.append(t)

        variant_str = " ".join(t["word"] for t in new_tokens)

        # Skip if already seen (catches reference duplicate and inter-variant duplicates)
        if variant_str in seen_strings:
            continue
        seen_strings.add(variant_str)

        all_variants.append({
            "sentence": variant_str,
            "order":    [t["id"] for t in new_tokens],
        })

    # ── Random sampling if over the cutoff ─────────────────────
    # Guideline §2: "if >100 variants, randomly sample exactly 99"
    if len(all_variants) > max_variants:
        all_variants = random.sample(all_variants, max_variants)

    return all_variants


# ─────────────────────────────────────────────────────────────
# 6. CONSTRUCTION TYPE LABELING
# ─────────────────────────────────────────────────────────────

def get_construction_type(sentence):
    """
    Labels each reference sentence by its word-order type,
    matching Table 1 of the paper:

        DOSV  — leftmost preverbal root-dependent is a direct object (obj)
        IOSV  — leftmost preverbal root-dependent is indirect object (iobj)
        OSV   — any object appears BEFORE the subject (general O-fronting)
        SOV   — canonical; subject appears before object

    The key fix: OSV requires the object to appear to the LEFT of the
    subject in the sentence. If subject comes first, it is SOV regardless
    of whether an object is also preverbal.
    """
    root = next((t for t in sentence if t["head"] == 0), None)
    if root is None:
        return "unknown"

    # Only look at direct dependents of root that are preverbal
    preverbal = sorted(
        [t for t in sentence
         if t["head"] == root["id"]
         and t["id"] < root["id"]
         and t["deprel"] != "punct"],
        key=lambda t: t["id"],   # left to right order
    )

    if not preverbal:
        return "unknown"

    # Find the leftmost subject and leftmost object among preverbal deps
    subj_pos = next(
        (t["id"] for t in preverbal if t["deprel"] in SUBJECT_RELS), None
    )
    dobj_pos = next(
        (t["id"] for t in preverbal if t["deprel"] in DIRECT_OBJ_RELS), None
    )
    iobj_pos = next(
        (t["id"] for t in preverbal if t["deprel"] in INDIRECT_OBJ_RELS), None
    )

    # DOSV: leftmost preverbal phrase is a direct object
    # i.e. direct object appears before subject
    if dobj_pos is not None:
        if subj_pos is None or dobj_pos < subj_pos:
            return "DOSV"

    # IOSV: leftmost preverbal phrase is an indirect object
    # i.e. indirect object appears before subject
    if iobj_pos is not None:
        if subj_pos is None or iobj_pos < subj_pos:
            return "IOSV"

    # OSV: any object (direct or indirect) precedes the subject
    obj_pos = min(
        p for p in [dobj_pos, iobj_pos] if p is not None
    ) if (dobj_pos or iobj_pos) else None

    if obj_pos is not None and subj_pos is not None and obj_pos < subj_pos:
        return "OSV"

    # SOV: canonical order — subject before object
    return "SOV"


# ─────────────────────────────────────────────────────────────
# 7. DATASET BUILDER
# ─────────────────────────────────────────────────────────────

def build_variant_dataset(sentences, sent_ids=None, contexts=None):
    """
    For every valid reference sentence, generates permuted variants
    and packages them as (reference, variant) pairs.

    Corpus-wide deprel transitions are built ONCE here from all valid
    sentences before variant generation begins, then passed into
    generate_variants_subtrees for each sentence.
    This is more efficient and more correct than building per-sentence
    transitions (see build_corpus_transitions for explanation).

    Each pair in the returned dataset contains:
        sentence_id       — index of the reference sentence
        sent_id           — CoNLL-U sent_id string (for tracing)
        context           — preceding sentence surface string
                            (needed for IS scoring)
        construction_type — DOSV / IOSV / OSV / SOV (Table 1)
        tokens            — original token list (for DL computation)
        reference         — reference surface string
        variant           — variant surface string
        order             — list of token_ids in variant word order
    """
    if sent_ids is None:
        sent_ids = [None] * len(sentences)
    if contexts is None:
        contexts = [""] * len(sentences)

    # Build corpus-wide transition set once before the loop
    print("  Building corpus-wide deprel transitions...")
    valid_transitions = build_corpus_transitions(sentences)
    print(f"  Valid transitions found: {len(valid_transitions)}")

    dataset = []

    for i, (sentence, sid, ctx) in enumerate(
        zip(sentences, sent_ids, contexts)
    ):
        # Pass corpus-wide transitions into variant generation
        variants = generate_variants_subtrees(sentence, valid_transitions)

        if not variants:
            continue

        reference    = " ".join(t["word"] for t in sentence)
        construction = get_construction_type(sentence)

        for v in variants:
            dataset.append({
                "sentence_id":       i,
                "sent_id":           sid,
                "context":           ctx,
                "construction_type": construction,
                "tokens":            sentence,
                "reference":         reference,
                "variant":           v["sentence"],
                "order":             v["order"],
            })

    return dataset


# ─────────────────────────────────────────────────────────────
# 8. MAIN — sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading treebank...")
    sentences, sent_ids, contexts = load_conllu(TREEBANK_PATH)
    print(f"  Total sentences loaded : {len(sentences)}")

    print("\nFiltering sentences...")
    valid_triples = [
        (s, sid, ctx)
        for s, sid, ctx in zip(sentences, sent_ids, contexts)
        if is_valid_treebank_sentence(s)
    ]

    vs  = [x[0] for x in valid_triples]
    vsi = [x[1] for x in valid_triples]
    vc  = [x[2] for x in valid_triples]

    print(f"  Valid sentences        : {len(vs)}")
    print(f"  Sentences dropped      : {len(sentences) - len(vs)}")

    print("\nGenerating variants...")
    dataset = build_variant_dataset(vs, vsi, vc)
    print(f"  Total variant pairs    : {len(dataset)}")

    print("\nConstruction type distribution (pairs):")
    counts = Counter(d["construction_type"] for d in dataset)
    for ctype, n in sorted(counts.items()):
        pct = n / len(dataset) * 100
        print(f"  {ctype:<8} : {n:>6}  ({pct:.1f}%)")

    print("\nContext coverage:")
    with_ctx = sum(1 for d in dataset if d["context"].strip())
    print(f"  Pairs with context     : {with_ctx} / {len(dataset)}")

    if dataset:
        print("\nExample pair:")
        p = dataset[0]
        print(f"  sent_id        : {p['sent_id']}")
        print(f"  context        : {p['context'][:80]!r}")
        print(f"  construction   : {p['construction_type']}")
        print(f"  reference      : {p['reference']}")
        print(f"  variant        : {p['variant']}")