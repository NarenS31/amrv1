"""
De-duplication with CORRECT distance threshold
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

print("="*80)
print("DE-DUPLICATION (FIXED)")
print("="*80)

# Load k-mers
X = np.load("data/combined/X_k6_unique.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

print(f"Original dataset: {len(genome_ids)} genomes")

# Normalize for cosine similarity
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

# Manual clustering: mark genomes to keep
print("\nFinding duplicates (this may take 10-20 min)...")

# Compute similarity matrix in chunks
similarity_threshold = 0.99
to_keep = set(range(len(genome_ids)))  # Start with all
removed_count = 0

# Check in chunks to save memory
chunk_size = 1000

for i in range(0, len(X_norm), chunk_size):
    print(f"Processing chunk {i//chunk_size + 1}/{(len(X_norm)//chunk_size)+1}...")
    
    chunk_end = min(i + chunk_size, len(X_norm))
    chunk = X_norm[i:chunk_end]
    
    # Compare this chunk to all PREVIOUS genomes we're keeping
    for j in range(i):
        if j not in to_keep:
            continue
        
        # Compare chunk to this genome
        sims = cosine_similarity(chunk, X_norm[j:j+1]).flatten()
        
        # Mark duplicates for removal (keep first occurrence)
        for k, sim in enumerate(sims):
            idx = i + k
            if idx in to_keep and sim > similarity_threshold and idx != j:
                to_keep.remove(idx)
                removed_count += 1

unique_indices = sorted(list(to_keep))

print(f"\nUnique genomes: {len(unique_indices)}")
print(f"Removed: {removed_count} duplicates")

if len(unique_indices) < 1000:
    print("\n⚠️ WARNING: Very few genomes remaining!")
    print("This suggests genome similarity is extremely high")
    print("Consider lowering threshold to 0.995 or 0.998")
    
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(0)

# Save
X_unique = X[unique_indices]
genome_ids_unique = [genome_ids[i] for i in unique_indices]

np.save("data/ml/X_k6_deduplicated.npy", X_unique)

with open("data/ml/genome_ids_deduplicated.txt", 'w') as f:
    for gid in genome_ids_unique:
        f.write(gid + '\n')

print("\n✓ Saved: data/ml/X_k6_deduplicated.npy")
print("✓ Saved: data/ml/genome_ids_deduplicated.txt")

# Re-extract embeddings
print("\n" + "="*80)
print("RE-EXTRACTING EMBEDDINGS")
print("="*80)

import torch
import sys
sys.path.append('src/src_nn')
from masked_autoencoder import MaskedAutoencoder
import pickle

with open("models/scaler_masked.pkl", 'rb') as f:
    scaler = pickle.load(f)

device = "cpu"
model = MaskedAutoencoder().to(device)
model.load_state_dict(torch.load("models/masked_ae_final.pt", map_location=device))
model.eval()

X_scaled = scaler.transform(X_unique.astype(np.float32))

print("Extracting embeddings...")
with torch.no_grad():
    Z_unique = model.encoder(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()

np.save("data/ml/Z_masked_deduplicated.npy", Z_unique)
print("✓ Saved: data/ml/Z_masked_deduplicated.npy")

print("\n" + "="*80)
print(f"Original: {len(genome_ids)}")
print(f"Deduplicated: {len(genome_ids_unique)}")
print(f"Reduction: {(1 - len(genome_ids_unique)/len(genome_ids))*100:.1f}%")
print("="*80)

