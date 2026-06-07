"""Run user-based CF on MovieLens 100k subset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collaborative_filtering import (
    load_ratings,
    mae,
    rmse,
    train_test_split,
    user_based_cf,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"


def main():
    ratings = load_ratings(DATA_DIR)
    # Small subset for quick demo (full 100k CF is slow without vectorization)
    ratings = ratings.sample(n=min(3000, len(ratings)), random_state=42)
    train, test = train_test_split(ratings, test_frac=0.1)
    test = test.head(200)  # cap predictions for demo runtime
    preds = user_based_cf(train, test, k=5)

    r = rmse(preds["rating"].values, preds["predicted"].values)
    m = mae(preds["rating"].values, preds["predicted"].values)
    print(f"Test size: {len(test)}")
    print(f"RMSE: {r:.4f}")
    print(f"MAE:  {m:.4f}")


if __name__ == "__main__":
    main()
