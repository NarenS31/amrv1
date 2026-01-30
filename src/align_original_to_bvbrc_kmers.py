"""
Re-extract original genomes using same k=6 format as BV-BRC
"""
import numpy as np
from pathlib import Path
from collections import Counter
from Bio import SeqIO
import gzip
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

# Generate all possible 6-mers


def generate_all_6mers():
    """Generate all 4^6 = 4096 possible 6-mers"""
    bases = ['A', 'C', 'G', 'T']
    kmers = []

    for b1 in bases:
        for b2 in bases:
            for b3 in bases:
                for b4 in bases:
                    for b5 in bases:
                        for b6 in bases:
                            kmers.append(b1+b2+b3+b4+b5+b6)

    return sorted(kmers)

# Function to extract k-mers from one genome


def extract_kmers_from_fasta(args):
    """Extract 6-mer counts from a genome"""
    idx, fasta_path, kmer_to_idx = args

    counts = np.zeros(4096, dtype=np.uint32)

    try:
        # Handle .gz files
        if str(fasta_path).endswith('.gz'):
            handle = gzip.open(fasta_path, 'rt')
        else:
            handle = open(fasta_path, 'r')

        for record in SeqIO.parse(handle, 'fasta'):
            seq = str(record.seq).upper()

            # Extract all k-mers
            for i in range(len(seq) - 5):
                kmer = seq[i:i+6]

                # Skip if contains N or other non-ACGT
                if all(b in 'ACGT' for b in kmer):
                    if kmer in kmer_to_idx:
                        counts[kmer_to_idx[kmer]] += 1

        handle.close()
        return idx, counts

    except Exception as e:
        print(f"Error processing {fasta_path}: {e}")
        return idx, np.zeros(4096, dtype=np.uint32)


def main():
    print("="*80)
    print("ALIGNING ORIGINAL GENOMES TO BV-BRC K-MER FORMAT")
    print("="*80)

    all_kmers = generate_all_6mers()
    kmer_to_idx = {k: i for i, k in enumerate(all_kmers)}

    print(f"Total k-mers: {len(all_kmers)}")
    assert len(all_kmers) == 4096

    # Find original genome files
    print("\n[1/3] Finding original genome files...")

    # Your original genomes should be in data/genomes/ncbi_genomes/
    genome_dir = Path("data/genomes/ncbi_genomes")

    if not genome_dir.exists():
        # Try alternative paths
        alternatives = [
            "data/raw/ncbi_genomes",
            "data/genomes/assemblies",
            "data/genomes/fna"
        ]

        for alt in alternatives:
            if Path(alt).exists():
                genome_dir = Path(alt)
                break
        else:
            print("ERROR: Cannot find original genome directory")
            print("Please specify the correct path")
            return

    print(f"Genome directory: {genome_dir}")

    # Find all FASTA files
    fasta_files = []
    for ext in ['*.fna', '*.fasta', '*.fa', '*.fna.gz', '*.fasta.gz']:
        fasta_files.extend(list(genome_dir.rglob(ext)))

    print(f"Found {len(fasta_files)} genome files")

    if len(fasta_files) == 0:
        print("ERROR: No genome files found!")
        return

    # Extract genome IDs from filenames
    genome_ids = []
    for f in fasta_files:
        # Extract GCF_XXXXXXXXX.X from filename
        fname = f.stem.replace('.fna', '').replace('.fasta', '')
        if fname.startswith('GCF_'):
            gid = '_'.join(fname.split('_')[:2])  # GCF_XXXXXXXXX.X
        else:
            gid = fname
        genome_ids.append(gid)

    print(f"Sample genome IDs: {genome_ids[:5]}")

    # Extract k-mers from all files
    print(f"\n[2/3] Extracting k-mers from {len(fasta_files)} genomes...")

    # Prepare arguments
    args_list = [(i, fpath, kmer_to_idx)
                 for i, fpath in enumerate(fasta_files)]

    # Use multiprocessing
    with Pool(processes=10) as pool:
        results = list(tqdm(
            pool.imap(extract_kmers_from_fasta, args_list),
            total=len(args_list),
            desc="Extracting k-mers"
        ))

    # Build matrix
    X_original = np.zeros((len(fasta_files), 4096), dtype=np.uint32)
    for idx, counts in results:
        X_original[idx] = counts

    print(f"\nExtracted matrix: {X_original.shape}")

    # Save
    print("\n[3/3] Saving...")

    np.save("data/genomes/X_original_k6.npy", X_original)
    print(f"✓ Saved: data/genomes/X_original_k6.npy {X_original.shape}")

    # Save genome IDs
    with open("data/genomes/genome_ids_original_k6.txt", 'w') as f:
        for gid in genome_ids:
            f.write(gid + '\n')

    print(
        f"✓ Saved: data/genomes/genome_ids_original_k6.txt ({len(genome_ids)} IDs)")

    # Verify
    print("\nVerification:")
    print(f"  Shape: {X_original.shape}")
    print(f"  Dtype: {X_original.dtype}")
    print(f"  Non-zero: {np.count_nonzero(X_original)}/{X_original.size}")
    print(f"  Memory: {X_original.nbytes / (1024**2):.1f} MB")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == '__main__':
    main()
