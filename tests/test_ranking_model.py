"""
test_ranking_model.py

Full evaluation suite for the pairwise ranking model:

    1. Train / test split evaluation  (80/20)
    2. Per-construction precision, recall, F1
    3. Confusion matrix per construction type
    4. Saves trained model weights to tests/output_test/ranking_model.pkl

Reference:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

This script goes beyond the paper's 10-fold CV (in train_ranking_model.py)
to provide a more detailed held-out evaluation.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

FEATURE_FILE = "data/features/features.csv"
MODEL_OUT    = "tests/output_test/ranking_model.pkl"
OUTPUT_PATH  = "tests/output_test/ranking_model_test.txt"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

FEATURES = ["delta_dl", "delta_trigram", "delta_lstm", "delta_adaptive", "delta_is"]
FEATURE_LABELS = {
    "delta_dl":       "Dependency Length",
    "delta_trigram":  "Trigram Surprisal",
    "delta_lstm":     "LSTM Surprisal",
    "delta_adaptive": "Adaptive Surprisal",
    "delta_is":       "Information Status",
}
CONSTRUCTION_TYPES = ["DOSV", "IOSV", "SOV"]


def pairwise_transform(df, feature_cols):
    """
    Joachims (2002) pairwise transformation.
    REF-VAR → label 1 (keep deltas as-is)
    VAR-REF → label 0 (negate all deltas)
    Returns balanced X, y arrays.
    """
    X_pos = df[feature_cols].values.copy()
    X_neg = -X_pos
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    return X, y


def sep(title="", w=62):
    if title:
        pad = (w - len(title) - 2) // 2
        return "=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2)
    return "=" * w


def evaluate_split(train_df, test_df, feature_cols, label=""):
    X_train, y_train = pairwise_transform(train_df, feature_cols)
    X_test,  y_test  = pairwise_transform(test_df,  feature_cols)

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X_train_z, y_train)

    preds = clf.predict(X_test_z)
    acc   = (preds == y_test).mean() * 100

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds)

    if label:
        print(f"\n  Accuracy  : {acc:.2f}%")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
        print(f"             Pred-0   Pred-1")
        print(f"  Actual-0 : {cm[0,0]:>6}   {cm[0,1]:>6}")
        print(f"  Actual-1 : {cm[1,0]:>6}   {cm[1,1]:>6}")

    return clf, scaler, acc, prec, rec, f1, preds, y_test


def main():

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"Loading: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)
    log(f"Shape  : {df.shape}")

    if "construction_type" not in df.columns:
        log("ERROR: construction_type column missing. Re-run build_feature_dataset.py")
        return

    avail = [f for f in FEATURES if f in df.columns]
    log(f"Features ({len(avail)}): {avail}")

    # ── Train / test split ─────────────────────────────────────
    log(sep("Train / Test Split"))
    unique_sents = df["sentence_id"].unique()
    train_sids, test_sids = train_test_split(
        unique_sents, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_df = df[df["sentence_id"].isin(train_sids)].copy()
    test_df  = df[df["sentence_id"].isin(test_sids)].copy()

    log(f"  Sentences — train: {len(train_sids)}, test: {len(test_sids)}")
    log(f"  Pairs     — train: {len(train_df)},  test: {len(test_df)}")

    # ── Full dataset evaluation ────────────────────────────────
    log(sep("Full dataset — all features"))
    clf_full, scaler_full, *_ = evaluate_split(
        train_df, test_df, avail, label="full"
    )

    # Coefficients — delta = ref - var, so sign convention matches paper directly:
    # negative coefficient = reference preferred (minimises the feature)
    log(f"\n  Regression coefficients (z-scored):")
    log(f"  (negative = reference preferred, matches paper Tables 2 & 3)")
    coefs = clf_full.coef_[0]
    for feat, coef in sorted(
        zip(avail, coefs), key=lambda x: -abs(x[1])
    ):
        direction = "↓ ref minimises" if coef < 0 else "↑ ref maximises"
        log(f"    {FEATURE_LABELS.get(feat, feat):<24}: {coef:+.4f}  ({direction})")

    # ── Per-construction evaluation ────────────────────────────
    log(sep("Per-construction evaluation"))
    summary_rows = []

    for ctype in CONSTRUCTION_TYPES:
        tr = train_df[train_df["construction_type"] == ctype]
        te = test_df[test_df["construction_type"] == ctype]

        if len(te) < 20:
            log(f"\n  {ctype}: insufficient test data ({len(te)} pairs) — skipping")
            continue

        log(f"\n  {ctype} — train: {len(tr)} pairs, test: {len(te)} pairs")

        _, _, acc, prec, rec, f1, preds, y_test = evaluate_split(
            tr, te, avail, label=ctype
        )

        log(f"\n  Classification report:")
        report = classification_report(
            y_test, preds,
            target_names=["VAR preferred", "REF preferred"],
            digits=4
        )
        for line in report.split("\n"):
            log("  " + line)

        summary_rows.append({
            "Construction": ctype,
            "Test pairs":   len(te) * 2,
            "Accuracy":     f"{acc:.2f}%",
            "Precision":    f"{prec:.4f}",
            "Recall":       f"{rec:.4f}",
            "F1":           f"{f1:.4f}",
        })

    # ── Individual predictor evaluation ───────────────────────
    log(sep("Individual predictor accuracy (test set)"))
    log(f"\n  {'Predictor':<26} {'Accuracy':>10} {'F1':>10}")
    log(f"  {'-'*26} {'-'*10} {'-'*10}")

    for feat in avail:
        _, _, acc, _, _, f1, _, _ = evaluate_split(train_df, test_df, [feat])
        log(f"  {FEATURE_LABELS.get(feat,feat):<26} {acc:>9.2f}% {f1:>10.4f}")

    base_feats = [f for f in ["delta_adaptive", "delta_trigram"] if f in avail]
    if len(base_feats) == 2:
        _, _, acc, _, _, f1, _, _ = evaluate_split(train_df, test_df, base_feats)
        log(f"  {'Baseline (adap+tri)':<26} {acc:>9.2f}% {f1:>10.4f}")

    _, _, acc, _, _, f1, _, _ = evaluate_split(train_df, test_df, avail)
    log(f"  {'All predictors':<26} {acc:>9.2f}% {f1:>10.4f}")

    # ── Summary table ──────────────────────────────────────────
    if summary_rows:
        log(sep("Summary"))
        summary = pd.DataFrame(summary_rows)
        log(summary.to_string(index=False))

    # ── Save model ─────────────────────────────────────────────
    log(sep("Saving model"))
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model_bundle = {
        "clf":       clf_full,
        "scaler":    scaler_full,
        "features":  avail,
        "test_size": TEST_SIZE,
    }
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model_bundle, f)
    log(f"  Saved to: {MODEL_OUT}")

    # ── Save output log ────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
