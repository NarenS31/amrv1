#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
from collections import defaultdict

SPLIT_CSV = "results/mash/genome_cluster_split.csv"
ALL_LIST  = "ALL_GENOMES_COMBINED.txt"          # paths to fasta files
PHENO_CSV = "data/bvbrc/phenotype.csv"
OUT_DIR   = "datasets/amr_by_drug"

MIN_SAMPLES = 1000

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def slug(s: str) -> str:
    s = norm(s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def basename_id(path: str) -> str:
    b = os.path.basename(path.strip())
    if b.endswith(".gz"): b = b[:-3]
    for suf in (".fna", ".fa", ".fasta"):
        if b.endswith(suf):
            b = b[:-len(suf)]
            break
    if b.endswith("_genomic"):
        b = b[:-len("_genomic")]
    return b

def extract_bvbrc_id(s: str):
    # BV-BRC genome IDs look like 573.14261
    m = re.search(r"(\d+\.\d+)", s)
    return m.group(1) if m else None

def main():
    for p in [SPLIT_CSV, ALL_LIST, PHENO_CSV]:
        if not os.path.exists(p):
            print(f"ERROR: missing required file: {p}")
            sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load split assignments (genome_id is derived from filename)
    splits = pd.read_csv(SPLIT_CSV)
    if not {"genome_id","split","cluster_id"}.issubset(set(splits.columns)):
        print("ERROR: split csv missing required columns:", splits.columns.tolist())
        sys.exit(2)

    splits["genome_id"] = splits["genome_id"].astype(str)

    # Build mapping: BV-BRC Genome ID -> exact genome_id used in splits
    # We do this using ALL_LIST -> basename -> match into splits
    split_ids = set(splits["genome_id"].tolist())

    bvbrc_to_splitid = {}
    collisions = defaultdict(list)

    with open(ALL_LIST, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            gid = basename_id(line)  # e.g. accn_573.14261.con.0023
            if gid not in split_ids:
                continue
            bid = extract_bvbrc_id(gid)  # 573.14261
            if not bid:
                continue
            collisions[bid].append(gid)

    # Resolve collisions deterministically: pick lexicographically smallest genome_id
    # (you could also pick longest/shortest; consistency matters more than rule)
    for bid, gids in collisions.items():
        bvbrc_to_splitid[bid] = sorted(set(gids))[0]

    print(f"BV-BRC IDs with at least one matching genome_id in splits: {len(bvbrc_to_splitid):,}")
    multi = sum(1 for bid,gids in collisions.items() if len(set(gids)) > 1)
    print(f"BV-BRC IDs that had >1 candidate genome_id (resolved by picking 1): {multi:,}")

    # Load phenotypes (BV-BRC)
    ph = pd.read_csv(PHENO_CSV, low_memory=False)
    need = {"Genome ID","Antibiotic","Resistant Phenotype","Evidence"}
    if not need.issubset(set(ph.columns)):
        print("ERROR: phenotype file missing columns:", need - set(ph.columns))
        print("Columns:", ph.columns.tolist())
        sys.exit(3)

    ph = ph[["Genome ID","Antibiotic","Resistant Phenotype","Evidence"]].copy()
    ph["Genome ID"] = ph["Genome ID"].astype(str)
    ph["Antibiotic"] = ph["Antibiotic"].astype(str)
    ph["Resistant Phenotype"] = ph["Resistant Phenotype"].astype(str)
    ph["Evidence"] = ph["Evidence"].astype(str)

    # Filters
    ph["evidence_norm"] = ph["Evidence"].map(norm)
    ph = ph[ph["evidence_norm"].str.contains("laboratory", na=False)].copy()

    ph["phenotype_norm"] = ph["Resistant Phenotype"].map(norm)
    ph = ph[ph["phenotype_norm"].isin({"resistant","susceptible"})].copy()

    # Map to exact split genome_id
    ph["genome_id"] = ph["Genome ID"].map(bvbrc_to_splitid)

    before = len(ph)
    ph = ph.dropna(subset=["genome_id"]).copy()
    print(f"Phenotype rows after filters: {before:,} -> after mapping to your genomes: {len(ph):,}")

    # Join split + cluster info (1-to-1 now)
    merged = ph.merge(splits, on="genome_id", how="left", validate="m:1")
    if merged["split"].isna().any():
        print("ERROR: some mapped genome_ids did not find a split. This should not happen.")
        bad = merged[merged["split"].isna()]["genome_id"].head(10).tolist()
        print("Examples:", bad)
        sys.exit(4)

    merged["antibiotic_norm"] = merged["Antibiotic"].map(norm)
    merged["y"] = merged["phenotype_norm"].map({"susceptible":0,"resistant":1}).astype(int)

    # Write per-drug datasets
    summary_rows = []
    for abx, d in merged.groupby("antibiotic_norm"):
        n = len(d)
        if n < MIN_SAMPLES:
            continue

        out_sub = os.path.join(OUT_DIR, slug(abx))
        os.makedirs(out_sub, exist_ok=True)

        for split in ["train","val","test"]:
            ds = d[d["split"]==split].copy()
            ds.to_csv(os.path.join(out_sub, f"{split}.csv"), index=False)

        r = int(d["y"].sum())
        s = n - r
        summary_rows.append({
            "antibiotic": abx,
            "n": n,
            "resistant": r,
            "susceptible": s,
            "resistant_pct": round(100*r/n, 2) if n else 0.0,
            "train_n": int((d["split"]=="train").sum()),
            "val_n": int((d["split"]=="val").sum()),
            "test_n": int((d["split"]=="test").sum()),
            "unique_genomes": int(d["genome_id"].nunique()),
        })

    summary = pd.DataFrame(summary_rows).sort_values("n", ascending=False)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"✓ Wrote per-drug datasets to: {OUT_DIR}")
    print(f"✓ Wrote summary: {summary_path}")
    print("\nTop 15 drugs by sample count:")
    print(summary.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
