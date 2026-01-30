#!/usr/bin/env python3
"""
Rebuild k-mer matrices using only deduplicated genomes
"""
import numpy as np
import pandas as pd
import pickle

print("="*80)
print("REBUILDING CLEAN DATASET")
print("="*80)

# Load keep list
with open('final_keep_list_gcf.txt') as f:
    keep_ids = set(line.strip() for line in f)

print(f"Genomes to keep: {len(keep_ids)}")

# Load original data
print("\nLoading original k-mer matrix...")
X_original = np.load("data/combined/X_k6_unique.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    original_ids = [line.strip().split('.')[0] for line in f]

print(f"Original matrix: {X_original.shape}")

# Find indices to keep
keep_indices = []
keep_genome_ids = []

for i, gid in enumerate(original_ids):
    if gid in keep_ids:
        keep_indices.append(i)
        keep_genome_ids.append(gid)

print(f"Matching genomes found: {len(keep_indices)}")

# Filter k-mers
X_clean = X_original[keep_indices]

print(f"Clean matrix shape: {X_clean.shape}")

# Save
np.save("data/ml/X_k6_clean.npy", X_clean)

with open("data/ml/genome_ids_clean.txt", 'w') as f:
    for gid in keep_genome_ids:
        f.write(gid + '\n')

print("\n✓ Saved: data/ml/X_k6_clean.npy")
print("✓ Saved: data/ml/genome_ids_clean.txt")

# Also need to extract clean embeddings if autoencoder was trained
print("\nExtracting embeddings for clean genomes...")

try:
    import torch
    import sys
    sys.path.append('src/src_nn')
    from masked_autoencoder import MaskedAutoencoder
    
    # Load scaler
    with open("models/scaler_masked.pkl", 'rb') as f:
        scaler = pickle.load()
    
    # Load model
    device = "cpu"
    model = MaskedAutoencoder().to(device)
    model.load_state_dict(torch.load("models/masked_ae_final.pt", map_location=device))
    model.eval()
    
    # Scale and extract
    X_scaled = scaler.transform(X_clean.astype(np.float32))
    
    with torch.no_grad():
        Z_clean = model.encoder(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()
    
    np.save("data/ml/Z_masked_clean.npy", Z_clean)
    print("✓ Saved: data/ml/Z_masked_clean.npy")
    
except Exception as e:
    print(f"⚠ Could not extract embeddings: {e}")
    print("  (You'll need to retrain autoencoder on clean data)")

# Filter phenotypes
print("\nFiltering phenotypes...")
pheno = pd.read_csv("data/merged/phenotypes_final.csv")
pheno_clean = pheno[pheno['genome_id'].isin(keep_ids)]

print(f"Original phenotypes: {len(pheno)}")
print(f"Clean phenotypes: {len(pheno_clean)}")

pheno_clean.to_csv("data/merged/phenotypes_clean.csv", index=False)
print("✓ Saved: data/merged/phenotypes_clean.csv")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Genomes: {len(keep_genome_ids)}")
print(f"Phenotype records: {len(pheno_clean)}")
print(f"Antibiotics: {pheno_clean['antibiotic'].nunique()}")
print("="*80)

