#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os

# Map base -> 2-bit code
BASE2 = np.full(256, 255, dtype=np.uint8)
BASE2[ord("A")] = 0
BASE2[ord("C")] = 1
BASE2[ord("G")] = 2
BASE2[ord("T")] = 3


def genome_id_from_filename(p: Path) -> str:
    name = p.name
    if name.endswith(".fna"):
        return name[:-4]
    if name.endswith(".fa"):
        return name[:-3]
    return name


def count_kmers_stream_fasta(path: Path, k: int) -> np.ndarray:
    """
    Stream FASTA file and count k-mers over A/C/G/T only.
    Uses rolling 2-bit encoding to index [0..4^k-1].
    Skips kmers that include non-ACGT (including N).
    """
    dim = 4 ** k
    counts = np.zeros(dim, dtype=np.uint32)

    mask = (1 << (2 * k)) - 1
    roll = 0
    valid_run = 0  # how many consecutive valid bases we’ve seen

    with path.open("r", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                # new contig resets the rolling window
                roll = 0
                valid_run = 0
                continue

            s = line.strip().upper()
            if not s:
                continue

            b = s.encode("ascii", "ignore")
            for ch in b:
                code = BASE2[ch]
                if code == 255:
                    roll = 0
                    valid_run = 0
                    continue
                roll = ((roll << 2) | int(code)) & mask
                valid_run += 1
                if valid_run >= k:
                    counts[roll] += 1

    return counts


def worker(args):
    i, path_str, k = args
    p = Path(path_str)
    gid = genome_id_from_filename(p)
    vec = count_kmers_stream_fasta(p, k)
    return i, gid, vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing .fna/.fa files (unzipped)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=6,
                    help="k-mer length (default: 6)")
    ap.add_argument("--n_jobs", type=int, default=8, help="Parallel workers")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N genomes (0=all)")
    args = ap.parse_args()

    if args.k <= 0 or args.k > 12:
        raise SystemExit("[ERROR] k must be 1..12 for sane memory/time.")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fna_files = sorted(
        [p for p in in_dir.iterdir()
         if p.is_file() and p.suffix.lower() in [".fna", ".fa"]]
    )
    if args.limit and args.limit > 0:
        fna_files = fna_files[:args.limit]

    n = len(fna_files)
    if n == 0:
        raise SystemExit(f"[ERROR] No .fna/.fa files found in {in_dir}")

    dim = 4 ** args.k
    print(f"Found {n} genomes in {in_dir}")
    print(f"Config: k={args.k} dim={dim} n_jobs={args.n_jobs}")

    # Outputs
    out_X = out_dir / f"X_k{args.k}.npy"
    out_ids = out_dir / f"genome_ids_k{args.k}.txt"

    # Create memmap-backed .npy (low RAM)
    X = np.lib.format.open_memmap(
        out_X, mode="w+", dtype=np.uint32, shape=(n, dim))

    work = [(i, str(p), args.k) for i, p in enumerate(fna_files)]
    genome_ids = [""] * n

    if args.n_jobs <= 1:
        for i, path_str, k in tqdm(work, total=n, desc="Counting k-mers"):
            ii, gid, vec = worker((i, path_str, k))
            X[ii, :] = vec
            genome_ids[ii] = gid
    else:
        ctx = mp.get_context("spawn")  # macOS-safe
        with ctx.Pool(processes=args.n_jobs) as pool:
            for ii, gid, vec in tqdm(pool.imap_unordered(worker, work, chunksize=8), total=n, desc="Counting k-mers"):
                X[ii, :] = vec
                genome_ids[ii] = gid

    # flush to disk
    X.flush()

    with out_ids.open("w") as f:
        for gid in genome_ids:
            f.write(gid + "\n")

    # quick sanity: how many nonzero kmers in first genome
    nnz0 = int(np.count_nonzero(X[0]))
    tot0 = int(X[0].sum())
    print(f"[DONE] wrote {out_X} shape={X.shape} dtype={X.dtype}")
    print(f"[DONE] wrote {out_ids} n={len(genome_ids)}")
    print(f"[SANITY] genome0 nnz={nnz0} sum_counts={tot0}")


if __name__ == "__main__":
    main()
