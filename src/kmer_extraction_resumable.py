#!/usr/bin/env python3
import argparse
import glob
import os
import time
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

# -------------------------
# FASTA streaming (no biopython)
# -------------------------


def iter_fasta_sequences(path: str):
    seq = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line[0] == ">":
                if seq:
                    yield "".join(seq)
                    seq = []
            else:
                seq.append(line.strip())
        if seq:
            yield "".join(seq)


# -------------------------
# k-mer hashing using rolling 2-bit encoding (A,C,G,T only)
# -------------------------
_MAP = np.full(256, -1, dtype=np.int8)
_MAP[ord("A")] = 0
_MAP[ord("C")] = 1
_MAP[ord("G")] = 2
_MAP[ord("T")] = 3
_MAP[ord("a")] = 0
_MAP[ord("c")] = 1
_MAP[ord("g")] = 2
_MAP[ord("t")] = 3


def extract_kmers_sparse(path: str, k: int, dim: int, max_bases: int | None = None):
    """
    Returns sparse hashed k-mer counts:
      idx: (nnz,) uint32
      cnt: (nnz,) uint32
    """
    mask = (1 << (2 * k)) - 1
    counts = {}  # python dict for sparse counting

    bases_seen = 0

    for seq in iter_fasta_sequences(path):
        if not seq:
            continue
        b = seq.encode("ascii", errors="ignore")
        code = 0
        valid = 0

        for ch in b:
            v = int(_MAP[ch])
            if v < 0:
                # break k-mer on N or weird chars
                code = 0
                valid = 0
                continue

            code = ((code << 2) | v) & mask
            valid += 1

            if valid >= k:
                idx = code % dim
                counts[idx] = counts.get(idx, 0) + 1

            bases_seen += 1
            if max_bases is not None and bases_seen >= max_bases:
                break

        if max_bases is not None and bases_seen >= max_bases:
            break

    if not counts:
        # Should basically never happen unless file is empty/invalid
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)

    idx = np.fromiter(counts.keys(), dtype=np.uint32, count=len(counts))
    cnt = np.fromiter(counts.values(), dtype=np.uint32, count=len(counts))

    # sort for nicer downstream
    order = np.argsort(idx)
    return idx[order], cnt[order]


def genome_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    # remove suffixes like _genomic.fna, .fna
    if base.endswith("_genomic.fna"):
        return base[:-len("_genomic.fna")]
    if base.endswith(".fna"):
        return base[:-len(".fna")]
    return os.path.splitext(base)[0]


def process_one(args):
    fna_path, out_dir, k, dim, max_bases = args
    gid = genome_id_from_filename(fna_path)

    out_per = os.path.join(out_dir, "per_genome")
    os.makedirs(out_per, exist_ok=True)

    out_path = os.path.join(out_per, f"{gid}.npz")
    if os.path.exists(out_path):
        return gid, True, 0, 0  # skipped

    tmp_path = out_path + ".tmp.npz"
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    t0 = time.time()
    idx, cnt = extract_kmers_sparse(
        fna_path, k=k, dim=dim, max_bases=max_bases)
    dt = time.time() - t0

    # sanity: if it's empty, treat as failure
    if idx.size == 0:
        raise RuntimeError(
            f"[BAD] produced empty features for {gid} from {fna_path}")

    np.savez_compressed(tmp_path, idx=idx, cnt=cnt,
                        dim=np.uint32(dim), k=np.uint32(k))
    os.replace(tmp_path, out_path)

    return gid, False, idx.size, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Folder with .fna files")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    ap.add_argument("--k", type=int, default=11)
    ap.add_argument("--dim", type=int, default=94826)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--max_bases", type=int, default=None,
                    help="Optional cap for debugging speed")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process first N files (debug)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_per = os.path.join(args.out_dir, "per_genome")
    os.makedirs(out_per, exist_ok=True)

    fna_files = sorted(glob.glob(os.path.join(args.input_dir, "*.fna")))
    if args.limit is not None:
        fna_files = fna_files[:args.limit]

    print(f"Found {len(fna_files)} .fna files in {args.input_dir}")
    print(f"Config: k={args.k} dim={args.dim} n_jobs={args.n_jobs}")

    work = [(p, args.out_dir, args.k, args.dim, args.max_bases)
            for p in fna_files]

    newly = 0
    skipped = 0
    total_nnz = 0
    times = []

    with Pool(processes=args.n_jobs) as pool:
        for gid, already, nnz, dt in tqdm(pool.imap_unordered(process_one, work), total=len(work)):
            if already:
                skipped += 1
            else:
                newly += 1
                total_nnz += nnz
                times.append(dt)

    print(f"[DONE] newly processed: {newly}  skipped(existing): {skipped}")
    if times:
        print(
            f"[STATS] avg_sec_per_genome={sum(times)/len(times):.3f}  avg_nnz={total_nnz/max(newly, 1):.1f}")
    print(f"[DONE] outputs in: {out_per}")


if __name__ == "__main__":
    main()
