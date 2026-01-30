#!/usr/bin/env python3
import os
import random
import pandas as pd

IN_SPLIT = "results/mash/genome_cluster_split.csv"   # has genome_id, cluster_id (and bad split)
OUT_SPLIT = "results/mash/genome_cluster_split.csv"
OUT_TRAIN = "results/mash/train_ids.txt"
OUT_VAL   = "results/mash/val_ids.txt"
OUT_TEST  = "results/mash/test_ids.txt"
OUT_SUM   = "results/mash/split_summary.txt"

SEED = 1337
FRAC_TRAIN = 0.80
FRAC_VAL   = 0.10
FRAC_TEST  = 0.10

def main():
    assert abs((FRAC_TRAIN+FRAC_VAL+FRAC_TEST)-1.0) < 1e-9

    df = pd.read_csv(IN_SPLIT)
    if not {"genome_id","cluster_id"}.issubset(df.columns):
        raise SystemExit(f"Missing columns in {IN_SPLIT}: {df.columns.tolist()}")

    # Work at cluster level to prevent leakage
    clusters = df[["cluster_id"]].drop_duplicates().copy()
    cluster_ids = clusters["cluster_id"].astype(str).tolist()

    random.seed(SEED)
    random.shuffle(cluster_ids)

    n = len(cluster_ids)
    n_train = int(n * FRAC_TRAIN)
    n_val   = int(n * FRAC_VAL)
    # ensure all clusters assigned
    n_test  = n - n_train - n_val

    # hard guards: never allow empty splits
    if n_train == 0:
        n_train = max(1, n - 2)
    if n_val == 0:
        n_val = 1
    if n_test == 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_clusters = set(cluster_ids[:n_train])
    val_clusters   = set(cluster_ids[n_train:n_train+n_val])
    test_clusters  = set(cluster_ids[n_train+n_val:])

    def assign(c):
        if c in train_clusters: return "train"
        if c in val_clusters:   return "val"
        return "test"

    df["cluster_id"] = df["cluster_id"].astype(str)
    df["split"] = df["cluster_id"].map(assign)

    # write outputs
    df.to_csv(OUT_SPLIT, index=False)

    for path, split in [(OUT_TRAIN,"train"),(OUT_VAL,"val"),(OUT_TEST,"test")]:
        ids = df.loc[df["split"]==split, "genome_id"].astype(str).tolist()
        with open(path, "w") as f:
            for x in ids:
                f.write(x + "\n")

    # summary
    with open(OUT_SUM, "w") as f:
        f.write(f"Clusters total: {n}\n")
        f.write(f"Clusters train: {len(train_clusters)}\n")
        f.write(f"Clusters val: {len(val_clusters)}\n")
        f.write(f"Clusters test: {len(test_clusters)}\n\n")
        for split in ["train","val","test"]:
            f.write(f"Genomes {split}: {(df['split']==split).sum()}\n")

    print("✓ Rebuilt splits (cluster-safe):")
    print(df["split"].value_counts())
    print(f"✓ Wrote: {OUT_SPLIT}")
    print(f"✓ Wrote: {OUT_TRAIN} {OUT_VAL} {OUT_TEST}")
    print(f"✓ Wrote: {OUT_SUM}")

if __name__ == "__main__":
    main()
