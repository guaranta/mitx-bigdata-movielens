"""Demo: k-mer PCA + k-means on synthetic DNA."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kmer_pca import kmer_frequencies, pca_kmeans, pca_plot_data

if __name__ == "__main__":
    np.random.seed(0)
    bases = "ACGT"
    seq = "".join(np.random.choice(list(bases), size=50000))
    matrix = kmer_frequencies(seq, kmer_len=4, window=5000)
    coords = pca_plot_data(matrix)
    proj, labels = pca_kmeans(matrix, n_clusters=3)
    print(f"Fragments: {matrix.shape[0]}, k-mers: {matrix.shape[1]}")
    print(f"PCA coords shape: {coords.shape}")
    print(f"Cluster labels: {np.bincount(labels)}")
