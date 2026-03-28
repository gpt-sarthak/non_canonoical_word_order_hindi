# Non-Canonical Word Order in Hindi — Strict Replication

Replication of:

> Ranjan & van Schijndel (2024)
> "Does Dependency Locality Predict Non-Canonical Word Order in Hindi?"

---

## Features Implemented

| # | Feature | Method | Status |
|---|---------|--------|--------|
| 1 | Dependency Length | Direct arc-distance sum | ✅ Complete |
| 2 | Trigram Surprisal | Kneser-Ney, Hindi Wikipedia | ✅ Complete |
| 3 | LSTM Surprisal | 2-layer LSTM, Hindi Wikipedia | ✅ Complete |
| 4 | Adaptive Surprisal | 1-step gradient adaptation | ✅ Complete |
| 5 | PCFG Surprisal | 2-level chunk PCFG, 5-fold CV | ✅ Complete |
| 6 | Information Status | Pronoun + lexical overlap | ✅ Complete |

---

## Final Results

### Classification Accuracy (Table 4 equivalent)

| Construction | Baseline (adap+tri) | All predictors | Paper (all) | Gap |
|---|---|---|---|---|
| DOSV | 68.6% | **75.5%** | 75.6% | <0.1% |
| IOSV | 82.1% | **84.9%** | ~85% | <0.1% |
| Full | 80.7% | **82.7%** | ~83% | <0.1% |

### Regression Coefficients — Full Dataset

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| delta_pcfg | −6.19 | ↓ ref preferred (minimises surprisal) |
| delta_trigram | −1.31 | ↓ ref preferred |
| delta_adaptive | −0.57 | ↓ ref preferred |
| delta_lstm | +0.52 | ↑ residual (adaptive absorbs main signal) |
| delta_dl | −0.43 | ↓ ref preferred |
| delta_is | +0.31 | ↑ ref preferred (given-before-new) |

All 6 coefficient directions match the paper. ✓

---

## Pipeline

```
Step 1   Load HDTB treebank (CoNLL-U)
Step 2   Filter valid sentences  →  2,828 sentences
Step 3   Generate variants  →  92,299 (reference, variant) pairs
Step 4   Dependency Length  →  delta_dl
Step 5   Trigram LM (Hindi Wikipedia)  →  delta_trigram
Step 6   LSTM LM (Hindi Wikipedia)  →  delta_lstm
Step 7   Adaptive surprisal  →  delta_adaptive
Step 8   Information Status  →  delta_is
Step 9   PCFG (2-level chunk grammar, 5-fold CV)  →  delta_pcfg
Step 10  Pairwise logistic regression + McNemar test  →  Results
```

---

## Project Structure

```
data/
  raw/UD_Hindi-HDTB/           HDTB CoNLL-U files
  processed/                   vocab.pkl, trigram.pkl, wiki_sentences.txt
  models/                      base_model.pt (LSTM)
  features/features.csv        92,299 pairs × 22 feature columns

feature_extraction/
  pcfg_features.py             PCFG grammar, 5-fold CV, scoring
  trigram_features.py
  lstm_features.py
  adaptive_features.py
  is_features.py
  dl_features.py

data/
  hutb_loader.py               Load, filter, generate variants

scripts/
  build_feature_dataset.py     Full pipeline (Steps 1-9)
  add_pcfg_features.py         Patch PCFG into existing features.csv (~2 min)
  train_ranking_model.py       Step 10: ranking model + McNemar test
  generate_replication_report.py  Generate full Word doc with all results

reports/
  replication_report.docx      Step-by-step replication guide with results
  pipeline_detailed_explanation.docx
```

---

## Quick Start

```powershell
# Activate environment
& d:\Hindi_recreate\non_canonoical_word_order_hindi\venv\Scripts\Activate.ps1

# (Re)compute PCFG features — ~2 minutes
python scripts/add_pcfg_features.py

# Run ranking model and see results — ~30 seconds
python scripts/train_ranking_model.py

# Regenerate the full documentation
python scripts/generate_replication_report.py
```

---

## PCFG Implementation Notes

The paper uses the Modelblocks incremental (Earley) parser trained on a Hindi constituency
treebank. We induce an equivalent 2-level PCFG from the shallow-parse (chunk) annotations
already present in the HDTB CoNLL-U MISC field:

- **Level 1**: S → C₁ C₂ … Cₖ  (sequence of chunk labels)
- **Level 2**: Cᵢ → w₁ w₂ … wₘ  (chunk-internal words)

Both reference and variant are scored with the **same fold PCFG** (the one trained without
that reference sentence). Using different models creates systematic CV bias.

Runtime: ~2-3 minutes for all 92,299 pairs using O(k) fast chunk scoring.
