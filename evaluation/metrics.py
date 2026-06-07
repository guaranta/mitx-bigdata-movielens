"""Clustering and recommender evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score


def clustering_metrics(features: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette and internal clustering quality."""
    if len(set(labels)) < 2:
        return {"silhouette": None, "n_clusters": len(set(labels))}
    return {
        "silhouette": float(silhouette_score(features, labels)),
        "n_clusters": len(set(labels)),
    }


def external_clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Adjusted Rand Index for external evaluation."""
    return {"adjusted_rand_index": float(adjusted_rand_score(true_labels, pred_labels))}


def recommender_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE and MAE for collaborative filtering."""
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    return {"rmse": rmse, "mae": mae}
