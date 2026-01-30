"""
Process downloaded genomes into k-mer matrix
"""
import numpy as np
from pathlib import Path
from collections import Counter
import itertools

def count_kmers(sequence, k=6):
    """Count k-mers in a sequence"""
    kmers = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if 'N' not in kmer:
            kmers[kmer] += 1
    return kmers

print("="*80)
print("PROCESSING 1,277 NEW GENOMES")
print("="*80)

# Generate all possible 6-mers (4^6 = 4096)
bases = ['A', 'C', 'G', 'T']
all_6mers = [''.join(kmer) for kmer in itertools.product(bases, repeat=6)]
kmer_to_idx = {kmer: i for i, kmer in enumerate(all_6mers)}
n_features = 4096

print(f"K-mer features: {n_features:,}")

# Find genome files
genome_dir = Path("data/ncbi_missing_fna")
genome_files = sorted(genome_dir.glob("*.fna"))

print(f"Genome files found: {len(genome_files):,}")

# Process genomes
X_new = []
genome_ids_new = []

for i, genome_file in enumerate(genome_files):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(genome_files)}")
    
    assembly_id = genome_file.stem
    
    try:
        # Read genome
        with open(genome_file, 'r') as f:
            sequence = ""
            for line in f:
                if not line.startswith('>'):
                    sequence += line.strip().upper()
        
        if len(sequence) < 100:
            continue
        
        # Count k-mers
        kmer_counts = count_kmers(sequence, k=6)
        
        # Convert to feature vector
        feature_vec = np.zeros(n_features, dtype=np.float32)
        for kmer, count in kmer_counts.items():
            if kmer in kmer_to_idx:
                feature_vec[kmer_to_idx[kmer]] = count
        
        # Normalize
        total = feature_vec.sum()
        if total > 0:
            feature_vec = feature_vec / total
        
        X_new.append(feature_vec)
        genome_ids_new.append(assembly_id)
        
    except Exception as e:
        print(f"  Error: {assembly_id}: {e}")
        continue

print(f"\n✓ Processed: {len(X_new):,} genomes")

# Convert to array
X_new = np.array(X_new, dtype=np.float32)

print(f"New k-mer matrix: {X_new.shape}")

# Save
np.save("data/ncbi_missing/X_k6_new.npy", X_new)

with open("data/ncbi_missing/genome_ids_new.txt", 'w') as f:
    for gid in genome_ids_new:
        f.write(gid + '\n')

print(f"\n✓ Saved: data/ncbi_missing/X_k6_new.npy")
print(f"✓ Saved: data/ncbi_missing/genome_ids_new.txt")

print("\n" + "="*80)

