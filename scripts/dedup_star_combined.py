#!/usr/bin/env python3
import gzip
import os
from collections import defaultdict

INFILE = "results/mash/near_dupes_combined_999.tsv.gz"
OUT_KEEP = "results/mash/keep_genomes_combined.txt"
OUT_INFO = "results/mash/dedup_combined_info.txt"

def canon_id(p: str) -> str:
    b = os.path.basename(p)
    for suf in [".fna", ".fa", ".fasta", ".gz"]:
        if b.endswith(suf):
            b = b[: -len(suf)]
    if b.endswith("_genomic"):
        b = b[: -len("_genomic")]
    parts = b.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else b

def main():
    hits = defaultdict(set)
    all_seen = set()
    edges = 0

    print("Reading edges...")
    with gzip.open(INFILE, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue

            a = canon_id(cols[0])
            b = canon_id(cols[1])
            hits[a].add(a)
            hits[a].add(b)
            all_seen.add(a)
            all_seen.add(b)
            edges += 1

            if edges % 500000 == 0:
                print(f"  {edges:,} edges...")

    print(f"Edges read: {edges:,}")
    print(f"Genomes seen: {len(all_seen):,}")
    print(f"Refs with edges: {len(hits):,}")

    kept = set()
    dropped = set()

    refs = sorted(hits.keys(), key=lambda r: len(hits[r]), reverse=True)
    for r in refs:
        if r in dropped:
            continue
        kept.add(r)
        for x in hits[r]:
            if x != r:
                dropped.add(x)

    with open(OUT_KEEP, "w") as f:
        for x in sorted(kept):
            f.write(x + "\n")

    with open(OUT_INFO, "w") as f:
        f.write(f"Input: {INFILE}\n")
        f.write(f"Edges read: {edges}\n")
        f.write(f"Genomes seen: {len(all_seen)}\n")
        f.write(f"Refs with edges: {len(hits)}\n")
        f.write(f"Kept: {len(kept)}\n")
        f.write(f"Dropped: {len(dropped)}\n")

    print(f"Kept: {len(kept):,}")
    print(f"Dropped: {len(dropped):,}")
    print("✓ Dedup complete")

if __name__ == "__main__":
    main()
