"""
Combine existing 20,528 genomes + new 1,277 genomes
"""
import numpy as np

print("="*80)
print("COMBINING ALL GENOMES")
print("="*80)

# Load existing matrix
X_existing = np.load("data/combined/X_k6_all.npy")
print(f"Existing genomes: {X_existing.shape}")

# Load new matrix
X_new = np.load("data/ncbi_missing/X_k6_new.npy")
print(f"New genomes: {X_new.shape}")

# Combine
X_combined = np.vstack([X_existing, X_new])
print(f"\nCombined: {X_combined.shape}")

# Load genome IDs
with open("data/combined/genome_ids_all.txt") as f:
    ids_existing = [line.strip() for line in f]

with open("data/ncbi_missing/genome_ids_new.txt") as f:
    ids_new = [line.strip() for line in f]

ids_combined = ids_existing + ids_new
print(f"Total genome IDs: {len(ids_combined):,}")

# Save
np.save("data/combined/X_k6_complete.npy", X_combined)

with open("data/combined/genome_ids_complete.txt", 'w') as f:
    for gid in ids_combined:
        f.write(gid + '\n')

print(f"\n✓ Saved: data/combined/X_k6_complete.npy ({X_combined.shape})")
print(f"✓ Saved: data/combined/genome_ids_complete.txt ({len(ids_combined):,} IDs)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total genomes: {X_combined.shape[0]:,}")
print(f"K-mer features: {X_combined.shape[1]:,}")
print(f"Memory: {X_combined.nbytes / (1024**3):.2f} GB")
print("="*80)

