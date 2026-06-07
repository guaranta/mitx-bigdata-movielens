"""Python port of MITx Week1 CalcFreq.m + PCAFreq.m + ClustFreq.m."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_fasta(path: Path) -> str:
    """Read FASTA sequence (concatenate all lines after headers)."""
    seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            seq.append(line.upper())
    return "".join(seq)


def kmer_frequencies(sequence: str, kmer_len: int = 4, window: int = 5000) -> np.ndarray:
    """Build k-mer frequency matrix from DNA fragments."""
    fragments = []
    i = 0
    while i + window <= len(sequence):
        fragments.append(sequence[i : i + window])
        i += window

    all_kmers: set[str] = set()
    frag_counts: list[Counter] = []
    for frag in fragments:
        counts: Counter = Counter()
        j = 0
        while j + kmer_len <= len(frag):
            counts[frag[j : j + kmer_len]] += 1
            j += kmer_len
        frag_counts.append(counts)
        all_kmers.update(counts.keys())

    kmer_list = sorted(all_kmers)
    matrix = np.zeros((len(fragments), len(kmer_list)))
    for i, counts in enumerate(frag_counts):
        for j, kmer in enumerate(kmer_list):
            matrix[i, j] = counts.get(kmer, 0)
    return matrix


def pca_plot_data(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Standardize and PCA-project (replaces PCAFreq.m)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(scaled)


def pca_kmeans(matrix: np.ndarray, n_clusters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """PCA + k-means (replaces ClustFreq.m)."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    pca = PCA(n_components=min(10, matrix.shape[1]))
    projected = pca.fit_transform(scaled)
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        projected
    )
    return projected[:, :2], labels
