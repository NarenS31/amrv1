"""
Remove duplicates from combined genome matrix
"""
import numpy as np

print("="*80)
print("DEDUPLICATING GENOMES")
print("="*80)

# Load matrix
X = np.load("data/combined/X_k6_complete.npy")
print(f"Current matrix: {X.shape}")

# Load IDs
with open("data/combined/genome_ids_complete.txt") as f:
    genome_ids = [line.strip() for line in f]

print(f"Total IDs: {len(genome_ids):,}")

# Find duplicates
seen = {}
unique_indices = []

for i, gid in enumerate(genome_ids):
    # Normalize ID (remove version)
    gid_norm = gid.split('.')[0]
    
    if gid_norm not in seen:
        seen[gid_norm] = i
        unique_indices.append(i)
    else:
        print(f"  Duplicate: {gid} (keeping first occurrence)")

print(f"\nUnique genomes: {len(unique_indices):,}")

# Keep only unique
X_unique = X[unique_indices]
genome_ids_unique = [genome_ids[i] for i in unique_indices]

print(f"Deduplicated matrix: {X_unique.shape}")

# Save
np.save("data/combined/X_k6_unique.npy", X_unique)

with open("data/combined/genome_ids_unique.txt", 'w') as f:
    for gid in genome_ids_unique:
        f.write(gid + '\n')

print(f"\n✓ Saved: data/combined/X_k6_unique.npy")
print(f"✓ Saved: data/combined/genome_ids_unique.txt")

print("\n" + "="*80)
print("FINAL DATASET")
print("="*80)
print(f"Unique genomes: {len(genome_ids_unique):,}")
print(f"K-mer features: {X_unique.shape[1]:,}")
print(f"Memory: {X_unique.nbytes / (1024**3):.2f} GB")
print("="*80)

