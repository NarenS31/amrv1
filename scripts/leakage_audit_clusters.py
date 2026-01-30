#!/usr/bin/env python3
import os
import pandas as pd

SPLIT_CSV = "results/mash/genome_cluster_split.csv"
OUT_DIR   = "results/cohort_v1"
OUT_TXT   = os.path.join(OUT_DIR, "leakage_audit.txt")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(SPLIT_CSV):
        raise SystemExit(f"❌ Missing: {SPLIT_CSV}")

    df = pd.read_csv(SPLIT_CSV)

    # flexible column handling
    cols = {c.lower(): c for c in df.columns}
    gid_col = cols.get("genome_id") or cols.get("genome") or cols.get("id")
    cl_col  = cols.get("cluster_id") or cols.get("cluster") or cols.get("clusterid")
    sp_col  = cols.get("split") or cols.get("set")

    if not (gid_col and cl_col and sp_col):
        raise SystemExit(f"❌ Could not infer required columns in {SPLIT_CSV}. Columns={df.columns.tolist()}")

    # Basic stats
    total = len(df)
    clusters = df[cl_col].nunique()
    split_counts = df[sp_col].value_counts().to_dict()

    # Core leakage test: each cluster must map to exactly one split
    cluster_split_n = df.groupby(cl_col)[sp_col].nunique()
    leaking_clusters = cluster_split_n[cluster_split_n > 1]

    # Cross-split overlap counts (should be 0)
    # Build sets of clusters per split
    split_sets = {s: set(df.loc[df[sp_col] == s, cl_col].unique()) for s in sorted(df[sp_col].unique())}
    splits = sorted(split_sets.keys())

    pair_overlaps = []
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            a, b = splits[i], splits[j]
            ov = len(split_sets[a].intersection(split_sets[b]))
            pair_overlaps.append((a, b, ov))

    # Write report
    with open(OUT_TXT, "w") as f:
        f.write("LEAKAGE AUDIT — CLUSTER-SAFE SPLITS\n")
        f.write(f"File: {SPLIT_CSV}\n\n")
        f.write(f"Rows (genomes): {total}\n")
        f.write(f"Unique clusters: {clusters}\n")
        f.write(f"Split counts: {split_counts}\n\n")

        f.write("Cluster appears in >1 split (should be 0):\n")
        f.write(f"  leaking_clusters_count = {int(len(leaking_clusters))}\n\n")

        f.write("Pairwise cluster overlap between splits (should all be 0):\n")
        for a, b, ov in pair_overlaps:
            f.write(f"  overlap({a},{b}) = {ov}\n")

        if len(leaking_clusters):
            f.write("\nTOP leaking clusters (first 20):\n")
            top = leaking_clusters.sort_values(ascending=False).head(20)
            for cid, n in top.items():
                splits_here = df.loc[df[cl_col] == cid, sp_col].unique().tolist()
                f.write(f"  {cid}: split_count={int(n)} splits={splits_here}\n")

    print(f"✓ wrote {OUT_TXT}")

    # Hard fail if leakage exists
    if len(leaking_clusters):
        raise SystemExit(f"❌ LEAKAGE DETECTED: {len(leaking_clusters)} clusters span multiple splits. Fix before presenting.")

    # Also hard fail if any overlap exists
    bad_pairs = [(a,b,ov) for a,b,ov in pair_overlaps if ov != 0]
    if bad_pairs:
        raise SystemExit(f"❌ LEAKAGE DETECTED: nonzero pairwise overlaps: {bad_pairs}")

    print("✓ PASS: no clusters span splits; pairwise overlaps are 0")

if __name__ == "__main__":
    main()
