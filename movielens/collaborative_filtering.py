"""User-based collaborative filtering on MovieLens 100k."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Load u.data: user_id, item_id, rating, timestamp."""
    path = data_dir / "u.data"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download MovieLens 100k — see data/README.md"
        )
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def train_test_split(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Random hold-out split."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(ratings)) < test_frac
    return ratings[~mask].copy(), ratings[mask].copy()


def user_based_cf(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 20,
) -> pd.DataFrame:
    """Predict ratings using k most similar users (Pearson correlation)."""
    user_item = train.pivot_table(
        index="user_id", columns="item_id", values="rating"
    )
    user_means = user_item.mean(axis=1)

    predictions = []
    for _, row in test.iterrows():
        uid, iid = row["user_id"], row["item_id"]
        if uid not in user_item.index or iid not in user_item.columns:
            pred = user_means.get(uid, train["rating"].mean())
            predictions.append(pred)
            continue

        target = user_item.loc[uid]
        rated = target.dropna()
        if len(rated) < 2:
            predictions.append(user_means[uid])
            continue

        sims = []
        for other_uid in user_item.index:
            if other_uid == uid:
                continue
            other = user_item.loc[other_uid]
            common = rated.index.intersection(other.dropna().index)
            if len(common) < 2:
                continue
            a = rated[common] - user_means[uid]
            b = other[common] - user_means[other_uid]
            denom = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
            if denom == 0:
                continue
            sims.append((other_uid, a.dot(b) / denom))

        sims.sort(key=lambda x: x[1], reverse=True)
        top_k = sims[:k]
        if not top_k:
            predictions.append(user_means[uid])
            continue

        num, den = 0.0, 0.0
        for other_uid, sim in top_k:
            val = user_item.loc[other_uid, iid]
            if pd.notna(val):
                num += sim * (val - user_means[other_uid])
                den += abs(sim)
        pred = user_means[uid] + (num / den if den else 0)
        predictions.append(pred)

    result = test.copy()
    result["predicted"] = predictions
    return result


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))
