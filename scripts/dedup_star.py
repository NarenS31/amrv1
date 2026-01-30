#!/usr/bin/env python3
import gzip
from collections import defaultdict

INFILE = "results/mash/near_dupes_999.tsv.gz"
OUT_KEEP = "results/mash/keep_genomes.txt"
OUT_DROP = "results/mash/drop_genomes.txt"
OUT_INFO = "results/mash/dedup_star_info.txt"

# Map ref -> set(nearby genomes)
hits = defaultdict(set)
all_seen = set()

print("Reading edges...")
with gzip.open(INFILE, "rt") as f:
    for i, line in enumerate(f):
        if i % 100000 == 0 and i > 0:
            print(f"  {i:,} edges...")
        parts = line.strip().split()
        a = parts[0].split('/')[-1].replace('.fna', '').replace('_genomic', '')
        b = parts[1].split('/')[-1].replace('.fna', '').replace('_genomic', '')
        
        # Extract GCF/GCA IDs
        a = '_'.join(a.split('_')[:2])
        b = '_'.join(b.split('_')[:2])
        
        hits[a].add(b)
        hits[a].add(a)
        all_seen.add(a)
        all_seen.add(b)

print(f"\nGenomes seen: {len(all_seen)}")
print(f"Reference genomes: {len(hits)}")

# Greedy: keep each ref, drop its neighbors
kept = set()
dropped = set()
clusters = []

refs = sorted(hits.keys(), key=lambda r: len(hits[r]), reverse=True)

for r in refs:
    if r in dropped:
        continue
    kept.add(r)
    for x in hits[r]:
        if x != r and x not in kept:
            dropped.add(x)
    clusters.append((r, len(hits[r])))

print(f"\nKept: {len(kept)}")
print(f"Dropped: {len(dropped)}")

with open(OUT_KEEP, "w") as f:
    for x in sorted(kept):
        f.write(x + "\n")

with open(OUT_DROP, "w") as f:
    for x in sorted(dropped):
        f.write(x + "\n")

with open(OUT_INFO, "w") as f:
    f.write(f"Genomes kept: {len(kept)}\n")
    f.write(f"Genomes dropped: {len(dropped)}\n")
    f.write(f"\nTop 20 refs by cluster size:\n")
    for r, sz in clusters[:20]:
        f.write(f"{sz}\t{r}\n")

print(f"\n✓ Saved: {OUT_KEEP}, {OUT_DROP}, {OUT_INFO}")

