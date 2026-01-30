#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
from multiprocessing import Pool, cpu_count
import xxhash

# ---------- FASTA streaming ----------
def iter_fasta(path):
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq).upper()
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            yield "".join(seq).upper()

def gid_from_path(p: str) -> str:
    b = os.path.basename(p.strip())
    if b.endswith(".gz"): b = b[:-3]
    for suf in (".fna",".fa",".fasta"):
        if b.endswith(suf):
            b = b[:-len(suf)]
            break
    if b.endswith("_genomic"):
        b = b[:-len("_genomic")]
    return b

def kmer_presence_hashed(path_k_dim):
    path, k, dim = path_k_dim
    vec = np.zeros(dim, dtype=np.uint8)

    # rolling window would be faster, but this is already a big improvement
    for seq in iter_fasta(path):
        L = len(seq)
        if L < k:
            continue
        for i in range(L - k + 1):
            kmer = seq[i:i+k]
            if "N" in kmer:
                continue
            h = xxhash.xxh64(kmer).intdigest() % dim
            vec[h] = 1
    return gid_from_path(path), vec

def process_split(list_file, k, dim, threads, outdir, split):
    paths = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)

    n = len(paths)
    X = np.zeros((n, dim), dtype=np.uint8)
    ids = [None] * n

    with Pool(processes=threads) as pool:
        for idx, (gid, vec) in enumerate(pool.imap_unordered(kmer_presence_hashed, [(p,k,dim) for p in paths], chunksize=4)):
            X_idx = idx
            X[X_idx] = vec
            ids[X_idx] = gid
            if (idx+1) % 200 == 0:
                print(f"{split}: {idx+1}/{n}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / f"X_{split}.npy", X)
    with open(outdir / f"ids_{split}.txt", "w") as f:
        for gid in ids:
            f.write(gid + "\n")

    print(f"✓ Saved {split}: {X.shape} -> {outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=31)
    ap.add_argument("--dim", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=min(10, cpu_count()))
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    process_split(args.train, args.k, args.dim, args.threads, args.outdir, "train")
    process_split(args.val,   args.k, args.dim, args.threads, args.outdir, "val")
    process_split(args.test,  args.k, args.dim, args.threads, args.outdir, "test")

if __name__ == "__main__":
    main()
