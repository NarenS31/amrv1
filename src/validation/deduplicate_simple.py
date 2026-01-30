"""
Simple de-duplication: random sampling to get diverse subset
Since 100% of pairs are >99% similar, we need aggressive sampling
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("="*80)
print("SIMPLE DE-DUPLICATION VIA SAMPLING")
print("="*80)

# Load data
X = np.load("data/combined/X_k6_unique.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

print(f"Original: {len(genome_ids)} genomes")

# Normalize
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

# Strategy: Iteratively select diverse genomes
# Start with a random genome, then keep adding genomes that are <99% similar to ALL selected ones

np.random.seed(42)
selected = [0]  # Start with first genome
threshold = 0.99

print("\nSelecting diverse genomes (this may take 30-60 min)...")

# Check in batches
batch_size = 100

for i in range(1, len(X), batch_size):
    if i % 1000 == 0:
        print(f"Processed {i}/{len(X)} genomes, selected {len(selected)} unique...")
    
    batch_end = min(i + batch_size, len(X))
    
    for j in range(i, batch_end):
        # Compare to ALL already selected genomes
        sims = cosine_similarity(X_norm[j:j+1], X_norm[selected]).flatten()
        
        # If dissimilar to ALL selected, add it
        if (sims < threshold).all():
            selected.append(j)
            
            # Early stop if we have enough
            if len(selected) >= 5000:
                break
    
    if len(selected) >= 5000:
        break

print(f"\nSelected {len(selected)} diverse genomes")

# Save
X_unique = X[selected]
genome_ids_unique = [genome_ids[i] for i in selected]

np.save("data/ml/X_k6_deduplicated.npy", X_unique)

with open("data/ml/genome_ids_deduplicated.txt", 'w') as f:
    for gid in genome_ids_unique:
        f.write(gid + '\n')

print("\n✓ Saved deduplicated data")

# Verify
X_unique_norm = X_unique / (np.linalg.norm(X_unique, axis=1, keepdims=True) + 1e-10)

# Check pairwise similarities in result
sample = min(100, len(X_unique))
sample_idx = np.random.choice(len(X_unique), sample, replace=False)

sim_matrix = cosine_similarity(X_unique_norm[sample_idx])
np.fill_diagonal(sim_matrix, 0)

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print(f"Max similarity among {sample} deduplicated genomes: {sim_matrix.max():.4f}")
print(f"Mean similarity: {sim_matrix.mean():.4f}")

if sim_matrix.max() < 0.99:
    print("✓ Deduplicated genomes are all <99% similar")
else:
    print("⚠️ Still have similar genomes")

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

with torch.no_grad():
    Z_unique = model.encoder(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()

np.save("data/ml/Z_masked_deduplicated.npy", Z_unique)
print("✓ Saved embeddings")

print("\n" + "="*80)
print(f"Original: {len(genome_ids)}")
print(f"Deduplicated: {len(genome_ids_unique)}")
print("="*80)

