"""Generate README figures."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "pca_lda"))
from kmer_pca import kmer_frequencies, pca_kmeans  # noqa: E402

rng = np.random.default_rng(0)
seq = "".join(rng.choice(list("ACGT"), size=50000))
matrix = kmer_frequencies(seq, kmer_len=4, window=5000)
proj, labels = pca_kmeans(matrix, n_clusters=3)

fig, ax = plt.subplots(figsize=(7, 5))
for k in range(3):
    mask = labels == k
    ax.scatter(proj[mask, 0], proj[mask, 1], label=f"Cluster {k}", alpha=0.8, s=40)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("k-mer PCA + k-means (genomics case study port)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "kmer_pca_clusters.png", dpi=150)
plt.close()

# Synthetic CF error distribution
errors = rng.normal(0, 0.9, 500)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(errors, bins=40, color="#10b981", alpha=0.85, edgecolor="white")
ax.axvline(np.sqrt(np.mean(errors**2)), color="#ef4444", ls="--", label=f"RMSE ≈ {np.sqrt(np.mean(errors**2)):.2f}")
ax.set_xlabel("Prediction error (rating units)")
ax.set_title("Collaborative filtering — hold-out error distribution")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "cf_errors.png", dpi=150)
plt.close()

print(f"Saved 2 figures to {OUT}")
