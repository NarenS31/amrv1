"""
Remove near-duplicate genomes using clustering
Keep one representative per cluster
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

print("="*80)
print("DE-DUPLICATING DATASET")
print("="*80)

# Load k-mers
X = np.load("data/combined/X_k6_unique.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

print(f"Original dataset: {len(genome_ids)} genomes")

# Normalize
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

# Cluster at 99% similarity threshold
# distance_threshold = 1 - 0.99 = 0.01
print("\nClustering genomes (this takes ~10-30 min for 21k genomes)...")
print("Using 99% similarity threshold (0.01 distance)")

clustering = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.01,  # 1 - 0.99 similarity
    metric='cosine',
    linkage='average'
)

labels = clustering.fit_predict(X_norm)

print(f"\nClusters found: {len(np.unique(labels))}")

# Keep one genome per cluster (first occurrence)
unique_indices = []
seen_clusters = set()

for i, label in enumerate(labels):
    if label not in seen_clusters:
        unique_indices.append(i)
        seen_clusters.add(label)

print(f"Unique genomes (after de-duplication): {len(unique_indices)}")
print(f"Removed: {len(genome_ids) - len(unique_indices)} duplicates")

# Save de-duplicated data
X_unique = X[unique_indices]
genome_ids_unique = [genome_ids[i] for i in unique_indices]

np.save("data/ml/X_k6_deduplicated.npy", X_unique)

with open("data/ml/genome_ids_deduplicated.txt", 'w') as f:
    for gid in genome_ids_unique:
        f.write(gid + '\n')

print("\n✓ Saved: data/ml/X_k6_deduplicated.npy")
print("✓ Saved: data/ml/genome_ids_deduplicated.txt")

# Also need to re-extract embeddings for de-duplicated set
print("\n" + "="*80)
print("RE-EXTRACTING EMBEDDINGS FOR DEDUPLICATED SET")
print("="*80)

import torch
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder
import pickle

# Load scaler and model
with open("models/scaler_masked.pkl", 'rb') as f:
    scaler = pickle.load(f)

device = "cpu"
model = MaskedAutoencoder().to(device)
model.load_state_dict(torch.load("models/masked_ae_final.pt", map_location=device))
model.eval()

# Scale and extract embeddings
X_scaled = scaler.transform(X_unique.astype(np.float32))

print("Extracting embeddings...")
with torch.no_grad():
    Z_unique = model.encoder(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()

np.save("data/ml/Z_masked_deduplicated.npy", Z_unique)
print("✓ Saved: data/ml/Z_masked_deduplicated.npy")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Original: {len(genome_ids)} genomes")
print(f"After de-duplication: {len(genome_ids_unique)} genomes")
print(f"Reduction: {(1 - len(genome_ids_unique)/len(genome_ids))*100:.1f}%")
print("\nNow re-run ALL experiments with deduplicated data!")
print("="*80)

