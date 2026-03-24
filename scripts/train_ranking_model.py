"""
train_ranking_model.py

Pairwise ranking logistic regression model with 10-fold cross-validation
and McNemar's significance testing.

Reference:
    Ranjan & van Schijndel (2024)
    "Does Dependency Locality Predict Non-canonical Word Order in Hindi?"

Paper methodology:
------------------
Ranking model (Eq. 4):
    choice ~ delta_adaptive + delta_trigram + delta_pcfg
             + delta_is + delta_dl

    where delta_x = feature_ref - feature_variant,
    all predictors normalised to z-scores.

Pairwise transformation (Joachims 2002):
    Each pair is duplicated with flipped features and flipped label.
    REF-VAR → label 1; VAR-REF → label 0.

Evaluation:
    10-fold cross-validation on the full transformed dataset.
    McNemar's two-tailed test to compare model pairs (Table 4).
    Separate regression analysis per construction type (Tables 2 & 3).

Notes:
    - PCFG surprisal is not available in this implementation
      (no constituency treebank for Hindi). The column is excluded.
      This matches the acknowledged limitation in the write-up.
    - IS score (delta_is) is included once is_features.py runs.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar


FEATURE_FILE = "data/features/features.csv"

# Features available in the current implementation
# (delta_pcfg excluded — not implemented, see module docstring)
ALL_FEATURES = ["delta_dl", "delta_trigram", "delta_lstm", "delta_adaptive", "delta_is"]

# Baseline: the two strongest individual predictors (paper Table 4)
BASELINE_FEATURES = ["delta_adaptive", "delta_trigram"]

# Construction types reported separately in the paper (Tables 2 & 3)
CONSTRUCTION_TYPES = ["DOSV", "IOSV", "OSV"]


# ------------------------------------------------------------
# pairwise_transform
#
# Converts (ref, var) pairs into a balanced binary dataset.
#
# Paper: "We generated ordered pairs comprising feature vectors
#  for both reference (REF) and variant (VAR) sentences,
#  maintaining a balance in the counts of each order type
#  (REF-VAR, VAR-REF). Pairs alternating between 'REF-VAR'
#  were labeled as '1,' whereas pairs in the 'VAR-REF' sequence
#  were labeled as '0.'"
#
# The delta columns already encode (ref - var), so:
#   REF-VAR row → keep deltas as-is  → label 1
#   VAR-REF row → negate all deltas  → label 0
# ------------------------------------------------------------
def pairwise_transform(df, feature_cols):

    X_pos = df[feature_cols].values.copy()   # REF-VAR, label 1
    X_neg = -X_pos                           # VAR-REF, label 0

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))

    return X, y


# ------------------------------------------------------------
# zscore_normalize
#
# Paper: "All the independent variables were normalised to
# z-scores."
# Fit scaler on training fold only; apply to test fold.
# ------------------------------------------------------------
def zscore_normalize(X_train, X_test):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z  = scaler.transform(X_test)
    return X_train_z, X_test_z


# ------------------------------------------------------------
# cv_accuracy
#
# 10-fold cross-validation returning per-fold predictions.
# Paper: "models trained on 9 folds of the dataset were used
# for prediction in the remaining fold."
# ------------------------------------------------------------
def cv_accuracy(X, y, n_splits=10, random_state=42):

    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds_all  = np.zeros(len(y), dtype=int)
    labels_all = y.copy()

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        X_train_z, X_test_z = zscore_normalize(X_train, X_test)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train_z, y_train)

        preds_all[test_idx] = clf.predict(X_test_z)

    accuracy = (preds_all == labels_all).mean() * 100
    return accuracy, preds_all


# ------------------------------------------------------------
# mcnemar_test
#
# Compares two sets of per-sample predictions.
# Paper: "McNemar's two-tailed significance test"
# Returns p-value.
# ------------------------------------------------------------
def mcnemar_test(preds_a, preds_b, y_true):

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency table
    n01 = np.sum( correct_a & ~correct_b)   # A correct, B wrong
    n10 = np.sum(~correct_a &  correct_b)   # A wrong, B correct

    table = [[np.sum(correct_a & correct_b), n01],
             [n10, np.sum(~correct_a & ~correct_b)]]

    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue


# ------------------------------------------------------------
# regression_coefficients
#
# Fits logistic regression on the full (z-scored) dataset and
# returns coefficients.  Used to produce the equivalent of
# Tables 2 and 3 in the paper.
#
# Paper: "for estimating regression coefficients in the model,
# we use the test data with transformed feature values as
# predictors for a given construction under study."
# ------------------------------------------------------------
def regression_coefficients(df, feature_cols):
    """
    Fits logistic regression on the full (z-scored) dataset.

    Regularisation:
        Paper uses R's glm() which has NO regularisation.
        We use C=1e6 (effectively no L2 penalty) to match.
        Do NOT use this function on the full dataset when adaptive
        and lstm are both present — they are correlated and will
        produce unstable coefficients without regularisation.
        Use per-construction subsets (DOSV, IOSV) as the paper does.

    Sign convention:
        delta = feature(reference) − feature(variant)  [ref − var]
        A negative coefficient means the model prefers the reference
        when it has a LOWER value — this matches paper Tables 2 & 3
        directly, with no additional sign flip required.
    """
    X_raw, y = pairwise_transform(df, feature_cols)

    scaler = StandardScaler()
    X_z    = scaler.fit_transform(X_raw)

    # C=1e6 matches paper's unregularised glm() — no L2 penalty
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
    clf.fit(X_z, y)

    return dict(zip(feature_cols, clf.coef_[0]))


# ------------------------------------------------------------
# classification_table
#
# Produces the equivalent of Table 4: individual and combined
# predictor accuracies on a given subset of the data.
# ------------------------------------------------------------
def classification_table(df, label, n_splits=10):

    print(f"\n{'='*60}")
    print(f"Construction: {label}  (pairs: {len(df)})")
    print(f"{'='*60}")

    available = [f for f in ALL_FEATURES if f in df.columns]

    # Available features (IS may be absent if not computed yet)
    if not available:
        print("  No feature columns found.")
        return

    print(f"\nIndividual predictors (random baseline = 50.0%):")

    # Store per-sample predictions for McNemar comparisons
    preds_store = {}
    X_full, y_full = pairwise_transform(df, available)

    for feat in available:
        X_f, y_f = pairwise_transform(df, [feat])
        acc, preds = cv_accuracy(X_f, y_f, n_splits=n_splits)
        preds_store[feat] = preds
        print(f"  {feat:<22}: {acc:.2f}%")

    # Baseline = adaptive + trigram
    base_feats = [f for f in BASELINE_FEATURES if f in df.columns]
    if len(base_feats) == 2:
        X_base, y_base = pairwise_transform(df, base_feats)
        base_acc, base_preds = cv_accuracy(X_base, y_base, n_splits=n_splits)
        preds_store["baseline"] = base_preds
        print(f"\nBaseline (adaptive + trigram): {base_acc:.2f}%")

        # Add predictors beyond baseline
        extra_feats = [f for f in available if f not in BASELINE_FEATURES]
        cumulative  = list(base_feats)

        for feat in extra_feats:
            cumulative.append(feat)
            X_c, y_c = pairwise_transform(df, cumulative)
            acc_c, preds_c = cv_accuracy(X_c, y_c, n_splits=n_splits)
            pvalue = mcnemar_test(preds_store["baseline"], preds_c, y_base)
            sig = "***" if pvalue < 0.001 else ("**" if pvalue < 0.01 else ("*" if pvalue < 0.05 else ""))
            print(f"  Baseline + {feat:<15}: {acc_c:.2f}%   p={pvalue:.4f} {sig}")
            preds_store[f"baseline+{feat}"] = preds_c

    # All predictors together
    X_all, y_all = pairwise_transform(df, available)
    acc_all, preds_all = cv_accuracy(X_all, y_all, n_splits=n_splits)
    print(f"  All predictors         : {acc_all:.2f}%")


# ------------------------------------------------------------
# regression_table
#
# Prints regression coefficients per construction type,
# matching Tables 2 and 3 of the paper.
# ------------------------------------------------------------
def regression_table(df, label):
    """
    Print regression coefficients matching Tables 2 and 3 of the paper.

    Sign convention (after Fix 1 negation in regression_coefficients):
        Negative coeff = reference preferred (ref has LOWER value)
        Positive coeff = variant preferred   (ref has HIGHER value)

    Paper Table 2 (DOSV): DL, trigram, adaptive all NEGATIVE
    → reference minimises surprisal and dependency length.
    IS score POSITIVE → reference has given-before-new order.
    """
    available = [f for f in ALL_FEATURES if f in df.columns]
    if not available:
        return

    print(f"\n--- Regression coefficients: {label} (N={len(df)}) ---")
    print(f"    (delta = ref − var; negative = reference preferred, matches paper)")
    coeffs = regression_coefficients(df, available)
    for feat, coef in sorted(coeffs.items(), key=lambda x: -abs(x[1])):
        if coef < 0:
            direction = "↓ ref preferred (minimises)"
        else:
            direction = "↑ ref preferred (maximises)"
        print(f"  {feat:<22}: {coef:+.4f}  ({direction})")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():

    print(f"Loading features from: {FEATURE_FILE}")
    df = pd.read_csv(FEATURE_FILE)

    print(f"Full dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "construction_type" not in df.columns:
        print("\nWARNING: 'construction_type' column missing.")
        print("Re-run build_feature_dataset.py to add it.")
        return

    # ----------------------------------------------------------
    # Full dataset — regression coefficients
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("FULL DATASET REGRESSION")
    print("="*60)
    regression_table(df, "Full dataset")

    # ----------------------------------------------------------
    # Per construction type — regression + classification
    # (Tables 2, 3, 4 in the paper)
    # ----------------------------------------------------------
    for ctype in CONSTRUCTION_TYPES:
        subset = df[df["construction_type"] == ctype].copy()
        if len(subset) == 0:
            print(f"\nNo pairs found for construction type: {ctype}")
            continue

        regression_table(subset, ctype)
        classification_table(subset, label=ctype)

    # ----------------------------------------------------------
    # Full dataset classification
    # ----------------------------------------------------------
    classification_table(df, label="Full dataset")


if __name__ == "__main__":
    main()