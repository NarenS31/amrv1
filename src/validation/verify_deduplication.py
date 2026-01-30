"""
Verify de-duplication results are correct
Sample random pairs and check their similarity
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("="*80)
print("VERIFYING DE-DUPLICATION")
print("="*80)

# Load original data
X_orig = np.load("data/combined/X_k6_unique.npy")
X_dedup = np.load("data/ml/X_k6_deduplicated.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    ids_orig = [line.strip() for line in f]

with open("data/ml/genome_ids_deduplicated.txt") as f:
    ids_dedup = [line.strip() for line in f]

print(f"Original: {len(ids_orig)} genomes")
print(f"Deduplicated: {len(ids_dedup)} genomes")
print(f"Reduction: {(1 - len(ids_dedup)/len(ids_orig))*100:.1f}%")

# Normalize
X_orig_norm = X_orig / (np.linalg.norm(X_orig, axis=1, keepdims=True) + 1e-10)
X_dedup_norm = X_dedup / (np.linalg.norm(X_dedup, axis=1, keepdims=True) + 1e-10)

# Check: Are deduplicated genomes actually different from each other?
print("\n" + "="*80)
print("CHECKING DEDUPLICATED SET")
print("="*80)

sample_size = min(100, len(X_dedup))
sample_indices = np.random.choice(len(X_dedup), sample_size, replace=False)

sim_matrix = cosine_similarity(X_dedup_norm[sample_indices])
np.fill_diagonal(sim_matrix, 0)

max_sim = sim_matrix.max()
mean_sim = sim_matrix[sim_matrix > 0].mean()

print(f"\nAmong {sample_size} deduplicated genomes:")
print(f"Max pairwise similarity: {max_sim:.4f}")
print(f"Mean pairwise similarity: {mean_sim:.4f}")

if max_sim > 0.99:
    print("\n⚠️ WARNING: Some 'unique' genomes are >99% similar!")
    print("De-duplication may have failed")
else:
    print("\n✓ Deduplicated genomes are diverse (all <99% similar)")

# Check: Were similar genomes actually removed from original?
print("\n" + "="*80)
print("CHECKING ORIGINAL DATASET")
print("="*80)

# Sample random 500 genomes from original
orig_sample = min(500, len(X_orig))
orig_indices = np.random.choice(len(X_orig), orig_sample, replace=False)

sim_matrix_orig = cosine_similarity(X_orig_norm[orig_indices])
np.fill_diagonal(sim_matrix_orig, 0)

# Count how many pairs are >99% similar
high_sim_count = (sim_matrix_orig > 0.99).sum() // 2  # Divide by 2 (symmetric)

print(f"\nAmong {orig_sample} random original genomes:")
print(f"Pairs with >99% similarity: {high_sim_count}")
print(f"Percentage of pairs: {high_sim_count / (orig_sample * (orig_sample-1) / 2) * 100:.1f}%")

if high_sim_count > orig_sample * 10:
    print("\n✓ Original dataset has MASSIVE duplication")
    print("  → 95% reduction is CORRECT")
    print("  → Your dataset genuinely has tons of near-identical genomes")
else:
    print("\n✗ Original dataset doesn't have that much duplication")
    print("  → De-duplication threshold may be too aggressive")

# Show some examples
print("\n" + "="*80)
print("EXAMPLE: Similar genomes in ORIGINAL dataset")
print("="*80)

# Find a highly similar pair
for i in range(min(100, len(X_orig))):
    for j in range(i+1, min(100, len(X_orig))):
        sim = cosine_similarity(X_orig_norm[i:i+1], X_orig_norm[j:j+1])[0, 0]
        if sim > 0.999:
            print(f"\n{ids_orig[i]} ↔ {ids_orig[j]}")
            print(f"Similarity: {sim:.6f} (>99.9%)")
            break
    else:
        continue
    break

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if high_sim_count > orig_sample * 5:
    print("✓ De-duplication appears CORRECT")
    print("  Your dataset genuinely has massive duplication")
    print("  This is common in outbreak surveillance datasets")
else:
    print("⚠️ De-duplication may be too aggressive")
    print("  Consider using 0.995 threshold instead of 0.99")

print("="*80)

