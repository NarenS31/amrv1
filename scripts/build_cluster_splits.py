#!/usr/bin/env python3
import csv
import gzip
import os
import random
from collections import defaultdict

EDGES_GZ = "results/mash/near_dupes_combined_999.tsv.gz"
ALL_LIST = "ALL_GENOMES_COMBINED.txt"

OUT_CSV = "results/mash/genome_cluster_split.csv"
OUT_TRAIN = "results/mash/train_ids.txt"
OUT_VAL = "results/mash/val_ids.txt"
OUT_TEST = "results/mash/test_ids.txt"
OUT_SUMMARY = "results/mash/split_summary.txt"

# Splits
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

SEED = 42

def genome_id_from_path(p: str) -> str:
    """
    Stable ID used for clustering/splitting.
    IMPORTANT: Do NOT truncate tokens (that causes collisions).
    """
    b = os.path.basename(p.strip())
    # strip compression and fasta suffixes
    if b.endswith(".gz"):
        b = b[:-3]
    for suf in (".fna", ".fa", ".fasta"):
        if b.endswith(suf):
            b = b[:-len(suf)]
            break
    if b.endswith("_genomic"):
        b = b[:-len("_genomic")]
    return b

class DSU:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x):
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # union by size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

def main():
    random.seed(SEED)

    # ---- Load all genomes (so singletons get assigned too) ----
    all_ids = []
    with open(ALL_LIST, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_ids.append(genome_id_from_path(line))

    all_ids_set = set(all_ids)
    if not all_ids:
        raise SystemExit("ERROR: ALL_GENOMES_COMBINED.txt is empty")

    # ---- Build DSU from edges ----
    dsu = DSU()
    for gid in all_ids_set:
        dsu.add(gid)

    edges = []
    edge_lines = 0

    with gzip.open(EDGES_GZ, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 2:
                continue
            a = genome_id_from_path(cols[0])
            b = genome_id_from_path(cols[1])
            # Only keep edges for genomes we actually know about
            if a in all_ids_set and b in all_ids_set:
                dsu.union(a, b)
                edges.append((a, b))
                edge_lines += 1

    # ---- Collect clusters (connected components) ----
    clusters = defaultdict(list)
    for gid in all_ids:
        clusters[dsu.find(gid)].append(gid)

    cluster_items = []
    for root, members in clusters.items():
        cluster_items.append((root, members))

    # Sort clusters by size descending (stabilizes split and avoids weird tiny cluster bias)
    cluster_items.sort(key=lambda x: len(x[1]), reverse=True)

    n_total = len(all_ids)
    target_train = int(TRAIN_FRAC * n_total)
    target_val = int(VAL_FRAC * n_total)
    target_test = n_total - target_train - target_val  # remainder

    # Greedy bin-pack: assign whole clusters to best split to approach targets
    split_members = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    targets = {"train": target_train, "val": target_val, "test": target_test}

    # Deterministic tie-breaking with seed: shuffle clusters of equal size
    i = 0
    while i < len(cluster_items):
        j = i
        sz = len(cluster_items[i][1])
        while j < len(cluster_items) and len(cluster_items[j][1]) == sz:
            j += 1
        block = cluster_items[i:j]
        random.shuffle(block)
        cluster_items[i:j] = block
        i = j

    cluster_id_map = {}
    split_map = {}

    # Make cluster ids stable-ish: just enumerate in sorted order
    for idx, (root, members) in enumerate(cluster_items, 1):
        cid = f"C{idx:07d}"
        cluster_id_map[root] = cid

        m = len(members)

        # choose split that keeps us closest to target after adding this cluster
        def score(split_name):
            after = split_counts[split_name] + m
            return abs(targets[split_name] - after)

        # Prefer train > val > test when scores tie (keeps train big)
        candidates = sorted(["train", "val", "test"], key=lambda s: (score(s), {"train":0,"val":1,"test":2}[s]))
        chosen = candidates[0]

        split_members[chosen].extend(members)
        split_counts[chosen] += m

        for g in members:
            split_map[g] = chosen

    # ---- Verify: no edge crosses splits ----
    crossings = 0
    for a, b in edges:
        if split_map.get(a) != split_map.get(b):
            crossings += 1
            if crossings <= 20:
                print("LEAKAGE EDGE (cross-split):", a, split_map.get(a), "<->", b, split_map.get(b))

    if crossings > 0:
        raise SystemExit(f"ERROR: Leakage detected. Crossing edges: {crossings}")

    # ---- Write outputs ----
    os.makedirs("results/mash", exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["genome_id", "cluster_id", "split"])
        for gid in all_ids:
            root = dsu.find(gid)
            w.writerow([gid, cluster_id_map[root], split_map[gid]])

    def write_list(path, ids):
        with open(path, "w") as f:
            for gid in sorted(ids):
                f.write(gid + "\n")

    write_list(OUT_TRAIN, split_members["train"])
    write_list(OUT_VAL, split_members["val"])
    write_list(OUT_TEST, split_members["test"])

    # Summaries
    n_clusters = len(cluster_items)
    largest = len(cluster_items[0][1]) if cluster_items else 0
    singles = sum(1 for _, m in cluster_items if len(m) == 1)

    with open(OUT_SUMMARY, "w") as f:
        f.write(f"Total genomes: {n_total}\n")
        f.write(f"Total clusters (connected components): {n_clusters}\n")
        f.write(f"Singleton clusters: {singles}\n")
        f.write(f"Largest cluster size: {largest}\n")
        f.write("\nSplit counts (genomes):\n")
        f.write(f"  train: {split_counts['train']}\n")
        f.write(f"  val:   {split_counts['val']}\n")
        f.write(f"  test:  {split_counts['test']}\n")
        f.write("\nEdges used for verification:\n")
        f.write(f"  edges_read: {edge_lines}\n")
        f.write("Leakage crossings: 0\n")

    print("✓ Wrote:", OUT_CSV)
    print("✓ Wrote:", OUT_TRAIN, OUT_VAL, OUT_TEST)
    print("✓ Wrote:", OUT_SUMMARY)
    print("✓ Verified: 0 leakage edges crossing splits")

if __name__ == "__main__":
    main()
