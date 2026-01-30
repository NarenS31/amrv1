"""
Combine original + BV-BRC k-mer matrices
"""
import os
import numpy as np
import pandas as pd

print("="*80)
print("COMBINING K-MER MATRICES")
print("="*80)

# Load both matrices
print("\n[1/4] Loading matrices...")

X_orig = np.load("data/genomes/X_original_k6.npy")
X_bvbrc = np.load("data/bvbrc/kmer_k6/X_k6.npy")

print(f"  Original:  {X_orig.shape}")
print(f"  BV-BRC:    {X_bvbrc.shape}")

# Load genome IDs
with open("data/genomes/genome_ids_original_k6.txt") as f:
    ids_orig = [line.strip() for line in f]

with open("data/bvbrc/kmer_k6/genome_ids_k6.txt") as f:
    ids_bvbrc = [line.strip() for line in f]

print(f"  Original IDs:  {len(ids_orig)}")
print(f"  BV-BRC IDs:    {len(ids_bvbrc)}")

# Check feature dimensions match
assert X_orig.shape[1] == X_bvbrc.shape[1], "Feature dimensions don't match!"
print(f"  ✓ Both have {X_orig.shape[1]} features")

# Find overlap
print("\n[2/4] Analyzing overlap...")

# Normalize genome IDs (remove suffixes)


def normalize_id(gid):
    # GCF_000364385.3_ASM36438v3_genomic -> GCF_000364385.3
    parts = gid.split('_')
    if len(parts) >= 2 and parts[0] == 'GCF':
        return f"{parts[0]}_{parts[1]}"
    return gid


ids_orig_norm = [normalize_id(x) for x in ids_orig]
ids_bvbrc_norm = [normalize_id(x) for x in ids_bvbrc]

orig_set = set(ids_orig_norm)
bvbrc_set = set(ids_bvbrc_norm)

overlap = orig_set.intersection(bvbrc_set)
only_orig = orig_set - bvbrc_set
only_bvbrc = bvbrc_set - orig_set

print(f"  Original only:   {len(only_orig):6d}")
print(f"  BV-BRC only:     {len(only_bvbrc):6d}")
print(f"  Overlap:         {len(overlap):6d}")
print(
    f"  TOTAL UNIQUE:    {len(only_orig) + len(only_bvbrc) + len(overlap):6d}")

# Combine: keep all unique genomes
# For overlap, use original version
print("\n[3/4] Combining matrices...")

# Keep all original
combined_X = [X_orig]
combined_ids = ids_orig_norm.copy()

# Add BV-BRC genomes not in original
for i, gid in enumerate(ids_bvbrc_norm):
    if gid not in orig_set:
        combined_X.append(X_bvbrc[i:i+1])
        combined_ids.append(gid)

# Stack
X_combined = np.vstack(combined_X)

print(f"  Combined matrix: {X_combined.shape}")
print(f"  Combined IDs:    {len(combined_ids)}")

# Save
print("\n[4/4] Saving combined dataset...")

os.makedirs("data/combined", exist_ok=True)

np.save("data/combined/X_k6_all.npy", X_combined)
print(f"  ✓ Saved: data/combined/X_k6_all.npy {X_combined.shape}")

with open("data/combined/genome_ids_all.txt", 'w') as f:
    for gid in combined_ids:
        f.write(gid + '\n')

print(f"  ✓ Saved: data/combined/genome_ids_all.txt ({len(combined_ids)} IDs)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total genomes:     {X_combined.shape[0]:6d}")
print(f"Features (k-mers): {X_combined.shape[1]:6d}")
print(f"Memory:            {X_combined.nbytes / (1024**3):.2f} GB")
print(
    f"Sparsity:          {np.count_nonzero(X_combined)/X_combined.size*100:.1f}% non-zero")
print("="*80)
