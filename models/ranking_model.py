import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_ranking_model(feature_file):
    """
    Train logistic regression ranking model.

    This follows the paper's pairwise ranking setup:
    ------------------------------------------------
    For each (reference, variant) pair:

    1. Use Δfeatures = variant − reference → label = 1
    2. Use -Δfeatures → label = 0

    This creates a balanced classification problem where the model
    learns which ordering is preferred.
    """

    df = pd.read_csv(feature_file)

    X = []
    y = []

    # ------------------------------------------------------------
    # Convert dataset into pairwise training format
    # ------------------------------------------------------------
    for _, row in df.iterrows():

        features = [
            row["delta_dl"],
            row["delta_trigram"],
            row["delta_lstm"],
            row["delta_adaptive"]
        ]

        # Case 1: reference preferred
        X.append(features)
        y.append(1)

        # Case 2: reverse (variant preferred)
        X.append([-f for f in features])
        y.append(0)

    # Convert to DataFrame
    X = pd.DataFrame(
        X,
        columns=[
            "delta_dl",
            "delta_trigram",
            "delta_lstm",
            "delta_adaptive"
        ]
    )

    y = pd.Series(y)

    # ------------------------------------------------------------
    # Train-test split
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------
    # Train logistic regression model
    # ------------------------------------------------------------
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print("\nRanking model accuracy:", acc)

    # ------------------------------------------------------------
    # Print feature coefficients (IMPORTANT for research)
    # ------------------------------------------------------------
    print("\nFeature coefficients:")

    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"{name}: {coef:.4f}")

    return model