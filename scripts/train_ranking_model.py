from models.ranking_model import train_ranking_model


FEATURE_FILE = "data/features/features.csv"


def main():

    print("Training ranking model...")

    model = train_ranking_model(FEATURE_FILE)

    print("Model training complete.")


if __name__ == "__main__":
    main()
