"""
adaptive_features.py

Computes adaptive LSTM surprisal features following:
    van Schijndel & Linzen (2018)
    "A Neural Model of Adaptation in Reading"

As applied in:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

─────────────────────────────────────────────────────────────
Paper methodology — Adaptive LSTM surprisal (§ Measures):
─────────────────────────────────────────────────────────────
"an LSTM language model is updated on each context sentence to
 better model the influence of discourse context (Example 1),
 and then surprisal of the words in the target sentences
 (Example 2) are estimated using the revised language model
 weights."

Correct procedure per UNIQUE REFERENCE SENTENCE:
    1. Reset model to base weights  (fresh for every new sentence)
    2. If a context sentence exists, do ONE gradient-update step
       on the CONTEXT sentence  (NOT the target itself)
    3. Score surprisal on the reference sentence   (eval mode)
    4. Score surprisal on every variant of that sentence (eval mode)
       — all variants share the same adapted weights
    5. delta_adaptive = surprisal_reference − surprisal_variant
       (paper convention: ref − var; negative = ref more predictable)

Key design decision — group by sentence_id:
    The dataset contains many (reference, variant) pairs that share
    the same reference sentence. Adapting once per pair would repeat
    the same gradient step N times for the same context, which is
    both wasteful and wrong. Instead we:
        - group all pairs by sentence_id
        - adapt ONCE per unique sentence
        - score all its variants under the same adapted weights

Bugs fixed vs original:
    1. Original adapted on the REFERENCE sentence itself — circular.
       Fixed: adapt on the CONTEXT sentence (preceding in document).
    2. Original accumulated weight updates across the whole dataset —
       by pair 1000 the model had drifted far from base.
       Fixed: reset to base_state at the start of every new sentence.
    3. Original set model.train() once at the top, leaving dropout
       active during surprisal scoring — non-deterministic.
       Fixed: eval() for scoring, train() only during adaptation step.
    4. Original adapted once per pair, not once per sentence.
       Fixed: group by sentence_id, adapt once, score all variants.
"""

import copy
from itertools import groupby

import torch
import torch.nn as nn

from feature_extraction.lstm_features import sentence_lstm_surprisal


# ─────────────────────────────────────────────────────────────
# _adapt_one_step
#
# Performs ONE gradient update on a given sentence string using
# cross-entropy next-word prediction loss.
#
# This simulates a reader updating their linguistic expectations
# after processing the preceding context sentence.
#
# Paper: "online fine-tuning of a neural language model"
# lr = 0.001 matches the Neural Complexity Toolkit default
# (van Schijndel & Linzen 2018).
# ─────────────────────────────────────────────────────────────
def _adapt_one_step(sentence_str, model, vocab, device, lr=0.01):
    """
    Run one SGD step on `sentence_str` to adapt the model.

    Skipped silently if the sentence has fewer than 2 tokens
    (cannot compute next-word prediction loss).

    Sets model to train() before the update and eval() after,
    so the caller always gets the model back in eval mode.
    """
    words = sentence_str.split()

    if len(words) < 2:
        # Need at least one input→target pair for next-word loss
        return

    indices = [vocab.get(w, vocab.get("<UNK>", 0)) for w in words]

    input_tensor  = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
    target_tensor = torch.tensor(indices[1:]).to(device)

    # Build a fresh optimizer each time so momentum/state doesn't
    # carry across sentences (pure SGD has no state, but explicit
    # per-call construction makes this guarantee clear).
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    # Train mode required for backward pass
    model.train()
    optimizer.zero_grad()

    logits, _ = model(input_tensor)
    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        target_tensor.view(-1)
    )
    loss.backward()
    optimizer.step()

    # Always restore eval mode after adaptation
    model.eval()


# ─────────────────────────────────────────────────────────────
# compute_adaptive_features
#
# Adds adaptive surprisal features to every item in the dataset.
#
# Input dataset items must contain:
#   "sentence_id"  — integer id shared by all pairs of one sentence
#   "reference"    — reference sentence string
#   "variant"      — variant sentence string
#   "context"      — preceding sentence string (empty = first in doc)
#
# Adds to each item:
#   "adaptive_reference"  — surprisal of reference (adapted weights)
#   "adaptive_variant"    — surprisal of variant   (adapted weights)
#   "delta_adaptive"      — adaptive_reference − adaptive_variant
#                           (paper convention: ref − var; negative = ref more predictable)
# ─────────────────────────────────────────────────────────────
def compute_adaptive_features(dataset, model, vocab, device):
    """
    Compute adaptive surprisal for all (reference, variant) pairs.

    Groups pairs by sentence_id so adaptation is performed exactly
    ONCE per unique reference sentence, not once per pair.
    All variants of a sentence are then scored under the same
    adapted model weights.
    """

    # ── Save base weights once ─────────────────────────────────
    # Every sentence starts from the same unmodified base model.
    # deepcopy is necessary — load_state_dict with the same dict
    # object would share tensor storage.
    base_state = copy.deepcopy(model.state_dict())
    model.eval()

    results = []

    # ── Group all pairs by sentence_id ────────────────────────
    # Dataset must be sorted by sentence_id for groupby to work.
    # We sort here defensively in case the caller didn't.
    sorted_dataset = sorted(dataset, key=lambda x: x["sentence_id"])

    total_sentences = 0
    total_pairs     = 0

    for sent_id, pair_group in groupby(
        sorted_dataset, key=lambda x: x["sentence_id"]
    ):
        pairs = list(pair_group)
        total_sentences += 1

        # All pairs in this group share the same reference and context
        reference = pairs[0]["reference"]
        context   = pairs[0].get("context", "")

        # ── Step 1: Reset to base weights ─────────────────────
        # Fresh start for every new sentence so updates from
        # previous sentences do not accumulate.
        model.load_state_dict(copy.deepcopy(base_state))
        model.eval()

        # ── Step 2: Adapt on the CONTEXT sentence ─────────────
        # Paper: "updated on each context sentence"
        # The context is the sentence that PRECEDED this one in
        # the document — retrieved from the treebank's sent_id
        # structure in hutb_loader.py.
        # If no context exists (first sentence in document),
        # we skip adaptation — matching the toolkit's behaviour.
        if context and context.strip():
            _adapt_one_step(context, model, vocab, device)

        # ── Step 3: Score reference (eval mode, dropout off) ──
        # Score once and reuse for all variants of this sentence.
        model.eval()
        s_ref = sentence_lstm_surprisal(reference, model, vocab, device)

        # ── Step 4: Score every variant (same adapted weights) ─
        for item in pairs:
            s_var = sentence_lstm_surprisal(
                item["variant"], model, vocab, device
            )

            # Paper: delta = feature(reference) − feature(variant)
            # Negative delta means reference is MORE predictable.
            delta = s_ref - s_var

            results.append({
                **item,
                "adaptive_reference": s_ref,
                "adaptive_variant":   s_var,
                "delta_adaptive":     delta,
            })
            total_pairs += 1

    # ── Restore base weights ───────────────────────────────────
    # Leave the model in its original state for any subsequent use.
    model.load_state_dict(base_state)
    model.eval()

    print(f"  Adaptive surprisal computed:")
    print(f"    Unique sentences processed : {total_sentences}")
    print(f"    Total pairs scored         : {total_pairs}")

    return results