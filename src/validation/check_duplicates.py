"""
Check for duplicate/near-duplicate genomes in dataset
Uses ANI (Average Nucleotide Identity) approximation via k-mer similarity
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pandas as pd

print("="*80)
print("DUPLICATE GENOME DETECTION")
print("="*80)

# Load k-mer matrix (not embeddings - need raw k-mers for similarity)
X = np.load("data/combined/X_k6_unique.npy")

with open("data/ml/genome_ids_all_complete.txt") as f:
    genome_ids = [line.strip().split('.')[0] for line in f]

print(f"Total genomes: {len(genome_ids)}")

# Normalize k-mer vectors for cosine similarity
X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

print("\nComputing pairwise similarities (this may take a few minutes)...")

# Sample for speed (check 1000 random genomes)
if len(X) > 1000:
    sample_indices = np.random.choice(len(X), 1000, replace=False)
    X_sample = X_norm[sample_indices]
    sample_ids = [genome_ids[i] for i in sample_indices]
    print(f"Sampling {len(sample_indices)} genomes for speed")
else:
    X_sample = X_norm
    sample_ids = genome_ids

# Compute pairwise cosine similarity
sim_matrix = cosine_similarity(X_sample)

# Set diagonal to 0 (ignore self-similarity)
np.fill_diagonal(sim_matrix, 0)

# Find high-similarity pairs
high_sim_threshold = 0.99  # ~99% ANI
very_high_sim_threshold = 0.995  # ~99.5% ANI

high_sim_pairs = []

for i in range(len(sim_matrix)):
    for j in range(i+1, len(sim_matrix)):
        if sim_matrix[i, j] > high_sim_threshold:
            high_sim_pairs.append({
                'genome_1': sample_ids[i],
                'genome_2': sample_ids[j],
                'similarity': sim_matrix[i, j]
            })

print("\n" + "="*80)
print("DUPLICATE DETECTION RESULTS")
print("="*80)

print(f"\nPairs with >99% similarity: {len([p for p in high_sim_pairs if p['similarity'] > 0.99])}")
print(f"Pairs with >99.5% similarity: {len([p for p in high_sim_pairs if p['similarity'] > 0.995])}")

if len(high_sim_pairs) > 0:
    print(f"\nTop 10 most similar pairs:")
    df_pairs = pd.DataFrame(high_sim_pairs).sort_values('similarity', ascending=False)
    print(df_pairs.head(10).to_string(index=False))
    
    df_pairs.to_csv("results/duplicate_genomes.csv", index=False)
    print("\n✓ Saved: results/duplicate_genomes.csv")
else:
    print("\n✓ No near-duplicate genomes found!")

# Check train/test leakage
print("\n" + "="*80)
print("CHECKING TRAIN/TEST LEAKAGE")
print("="*80)

# Load phenotypes to create train/test split
pheno = pd.read_csv("data/merged/phenotypes_final.csv")
genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}

# Get labeled genomes
labeled_indices = []
for gid in pheno['genome_id'].unique():
    if gid in genome_to_idx:
        labeled_indices.append(genome_to_idx[gid])

if len(labeled_indices) > 100:
    # Sample for speed
    labeled_sample = np.random.choice(labeled_indices, 100, replace=False)
else:
    labeled_sample = labeled_indices

# Create train/test split
train_idx, test_idx = train_test_split(
    labeled_sample, test_size=0.2, random_state=42
)

print(f"Train samples: {len(train_idx)}")
print(f"Test samples: {len(test_idx)}")

# Check if any train genome is too similar to any test genome
X_train = X_norm[train_idx]
X_test = X_norm[test_idx]

cross_sim = cosine_similarity(X_test, X_train)
max_cross_sim = cross_sim.max(axis=1)

leakage_count = (max_cross_sim > 0.99).sum()

print(f"\nTest genomes with >99% similarity to ANY train genome: {leakage_count}/{len(test_idx)}")

if leakage_count > 0:
    print("\n⚠️ WARNING: Potential data leakage detected!")
    print("Some test genomes are nearly identical to training genomes")
    print("This inflates performance estimates")
    
    leak_indices = np.where(max_cross_sim > 0.99)[0]
    print(f"\nMost similar test genomes:")
    for idx in leak_indices[:5]:
        test_gid = genome_ids[test_idx[idx]]
        train_match_idx = cross_sim[idx].argmax()
        train_gid = genome_ids[train_idx[train_match_idx]]
        similarity = cross_sim[idx, train_match_idx]
        print(f"  Test: {test_gid} ↔ Train: {train_gid} (sim={similarity:.4f})")
else:
    print("\n✓ No train/test leakage detected in sampled genomes")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if len(high_sim_pairs) > 10:
    print("⚠️ Multiple near-duplicate genomes found")
    print("  → Consider removing duplicates before training")
    print("  → Or use clustering to ensure train/test independence")
elif leakage_count > 0:
    print("⚠️ Train/test leakage detected")
    print("  → Re-split data ensuring >99% similar genomes are in same fold")
else:
    print("✓ Dataset appears clean of duplicates and leakage")

print("="*80)

