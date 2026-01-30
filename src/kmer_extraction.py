# import os
# import numpy as np
# from tqdm import tqdm

# # base folder with GCF_XXX folders
# GENOME_DIR = "data/genomes/ncbi_genomes/ncbi_dataset/data"
# KMER_SIZE = 6  # example k-mer size, change if needed


# def find_fna_files(base_dir):
#     """Recursively find all .fna files under base_dir"""
#     fna_files = []
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith(".fna"):
#                 fna_files.append(os.path.join(root, file))
#     return fna_files


# def read_fasta(file_path):
#     """Read sequences from a .fna FASTA file"""
#     seqs = []
#     with open(file_path, "r") as f:
#         sequence = ""
#         for line in f:
#             line = line.strip()
#             if line.startswith(">"):
#                 if sequence:
#                     seqs.append(sequence)
#                     sequence = ""
#             else:
#                 sequence += line
#         if sequence:
#             seqs.append(sequence)
#     return seqs


# def kmer_counts(sequence, k=KMER_SIZE):
#     """Return k-mer counts for a sequence"""
#     counts = {}
#     for i in range(len(sequence) - k + 1):
#         kmer = sequence[i:i+k]
#         counts[kmer] = counts.get(kmer, 0) + 1
#     return counts


# def extract_kmer_features(fna_files):
#     all_features = []
#     for fna_file in tqdm(fna_files, desc="Processing genomes"):
#         sequences = read_fasta(fna_file)
#         genome_features = {}
#         for seq in sequences:
#             counts = kmer_counts(seq)
#             # merge counts into genome_features
#             for kmer, c in counts.items():
#                 genome_features[kmer] = genome_features.get(kmer, 0) + c
#         all_features.append(genome_features)
#     return all_features


# if __name__ == "__main__":
#     fna_files = find_fna_files(GENOME_DIR)
#     print(f"Found {len(fna_files)} .fna files")
#     kmer_features = extract_kmer_features(fna_files)
#     np.save("data/genomes/kmer_features.npy", kmer_features)
#     print(f"Saved k-mer features: data/genomes/kmer_features.npy")
#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def read_fna_text(path: Path) -> str:
    # FASTA: drop header lines, concat sequences
    seq_parts = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                continue
            seq_parts.append(line.strip().upper())
    return "".join(seq_parts)


def kmer_counts(seq: str, k: int) -> dict:
    # Basic k-mer counter. Ignores kmers with non-ACGT.
    c = Counter()
    n = len(seq)
    if n < k:
        return {}
    for i in range(n - k + 1):
        kmer = seq[i: i + k]
        # fast validity check
        if "N" in kmer:
            continue
        ok = True
        for ch in kmer:
            if ch not in "ACGT":
                ok = False
                break
        if not ok:
            continue
        c[kmer] += 1
    return dict(c)


def process_one(args):
    path_str, k = args
    p = Path(path_str)
    # genome_id = filename without extension(s)
    genome_id = p.name
    if genome_id.endswith(".fna"):
        genome_id = genome_id[:-4]
    elif genome_id.endswith(".fa"):
        genome_id = genome_id[:-3]
    # read + count
    seq = read_fna_text(p)
    feats = kmer_counts(seq, k)
    return genome_id, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing .fna files (unzipped)")
    ap.add_argument("--out_dir", required=True,
                    help="Directory to write outputs")
    ap.add_argument("--k", type=int, default=8, help="k-mer length")
    ap.add_argument("--n_jobs", type=int, default=8, help="Parallel workers")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise SystemExit(f"[ERROR] input_dir does not exist: {in_dir}")

    fna_files = sorted([p for p in in_dir.iterdir() if p.is_file()
                       and p.suffix.lower() in [".fna", ".fa"]])

    print(f"Found {len(fna_files)} .fna files in {in_dir}")

    # multiproc
    work = [(str(p), args.k) for p in fna_files]

    genome_ids = []
    features = []

    if args.n_jobs <= 1:
        for item in tqdm(work, total=len(work), desc="Processing genomes"):
            gid, feats = process_one(item)
            genome_ids.append(gid)
            features.append(feats)
    else:
        ctx = mp.get_context("spawn")  # safer on macOS
        with ctx.Pool(processes=args.n_jobs) as pool:
            for gid, feats in tqdm(pool.imap_unordered(process_one, work), total=len(work), desc="Processing genomes"):
                genome_ids.append(gid)
                features.append(feats)

        # keep deterministic order (so ids align with features)
        order = np.argsort(np.array(genome_ids))
        genome_ids = [genome_ids[i] for i in order]
        features = [features[i] for i in order]

    out_feats = out_dir / "kmer_features.npy"
    out_ids = out_dir / "genome_ids.txt"

    np.save(out_feats, np.array(features, dtype=object))
    with out_ids.open("w") as f:
        for gid in genome_ids:
            f.write(gid + "\n")

    print(f"Saved k-mer features to {out_feats}")
    print(f"Saved genome IDs to {out_ids}")


if __name__ == "__main__":
    main()
