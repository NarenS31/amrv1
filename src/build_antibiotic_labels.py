#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def slugify(name: str) -> str:
    """Safe filename slug for antibiotic names."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_pheno(x: str) -> str:
    """Normalize phenotype strings to: resistant/susceptible/intermediate/unknown."""
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()

    # common values
    if s in {"r", "resistant"}:
        return "resistant"
    if s in {"s", "susceptible"}:
        return "susceptible"
    if s in {"i", "intermediate"}:
        return "intermediate"

    # phrases
    if "resist" in s:
        return "resistant"
    if "suscept" in s:
        return "susceptible"
    if "intermed" in s:
        return "intermediate"

    return "unknown"


def resolve_duplicate_labels(labels):
    """
    Given multiple phenotype strings for the same (BioSample, Antibiotic),
    return a single binary label or None if unusable.

    Rule:
      - if any resistant -> 1
      - else if any susceptible -> 0
      - else -> None (intermediate/unknown only)
    """
    normalized = [normalize_pheno(x) for x in labels]
    if "resistant" in normalized:
        return 1
    if "susceptible" in normalized:
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmer_matrix", default="data/genomes/kmer_matrix.npy")
    ap.add_argument("--genome_ids", default="data/genomes/genome_ids.txt")
    ap.add_argument("--mapping_csv",
                    default="data/genomes/genome_to_biosample.csv")
    ap.add_argument(
        "--phenos_csv", default="data/amr_tables/phenotypes_clean.csv")
    ap.add_argument("--out_dir", default="data/ml")

    # Filtering thresholds
    ap.add_argument("--min_samples", type=int, default=200,
                    help="Minimum labeled genomes (R+S) required to keep an antibiotic.")
    ap.add_argument("--min_pos", type=int, default=30,
                    help="Minimum resistant genomes required to keep an antibiotic.")
    ap.add_argument("--min_neg", type=int, default=30,
                    help="Minimum susceptible genomes required to keep an antibiotic.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ----------------------------
    # Load genome IDs (kmer order)
    # ----------------------------
    if not os.path.exists(args.genome_ids):
        raise FileNotFoundError(f"Missing genome_ids.txt: {args.genome_ids}")

    with open(args.genome_ids) as f:
        genome_ids_full = [line.strip() for line in f if line.strip()]

    if len(genome_ids_full) == 0:
        raise RuntimeError(
            "genome_ids.txt is empty. Re-check your kmer extraction step.")

    # genome_ids.txt entries like: GCF_029081955.1_ASM2908195v1 -> shorten to GCF_029081955.1
    genome_ids_short = ["_".join(g.split("_")[:2]) for g in genome_ids_full]
    genome_to_index = {gid: i for i, gid in enumerate(genome_ids_short)}

    # ----------------------------
    # Load mapping + phenotypes
    # ----------------------------
    mapping = pd.read_csv(args.mapping_csv)
    phenos = pd.read_csv(args.phenos_csv)

    required_pheno_cols = {"BioSample", "Antibiotic", "Resistance phenotype"}
    missing_cols = required_pheno_cols - set(phenos.columns)
    if missing_cols:
        raise KeyError(f"phenotypes_clean.csv is missing columns: {missing_cols}. "
                       f"Found: {list(phenos.columns)}")

    if "genome_id" not in mapping.columns or "BioSample" not in mapping.columns:
        raise KeyError(
            "genome_to_biosample.csv must contain columns ['genome_id','BioSample']. "
            f"Found: {list(mapping.columns)}"
        )

    # Keep only genomes that exist in kmer order
    mapping = mapping[mapping["genome_id"].isin(genome_to_index)].copy()

    # Merge to connect genome_id <-> BioSample with phenotype rows
    merged = mapping.merge(phenos, on="BioSample", how="inner")

    # ----------------------------
    # Resolve duplicates per (BioSample, Antibiotic)
    # ----------------------------
    grouped = (
        merged.groupby(["BioSample", "Antibiotic"])["Resistance phenotype"]
        .apply(list)
        .reset_index()
    )

    grouped["label"] = grouped["Resistance phenotype"].apply(
        resolve_duplicate_labels)

    # Drop unusable rows (intermediate/unknown-only)
    grouped = grouped.dropna(subset=["label"]).copy()
    grouped["label"] = grouped["label"].astype(int)

    # ----------------------------
    # Map BioSample -> genome_id (choose one genome per BioSample)
    # ----------------------------
    biosample_to_genome = mapping.drop_duplicates(
        "BioSample")[["BioSample", "genome_id"]].copy()
    grouped = grouped.merge(biosample_to_genome, on="BioSample", how="inner")

    grouped["kmer_index"] = grouped["genome_id"].map(genome_to_index)

    # ----------------------------
    # Stats + DEBUG PRINT (before filtering)
    # ----------------------------
    stats = grouped.groupby("Antibiotic")["label"].agg(["count", "sum"])
    stats["neg"] = stats["count"] - stats["sum"]
    stats = stats.sort_values("count", ascending=False)

    print("\n=== DEBUG LABEL STATS (before filtering) ===")
    print("Merged rows:", merged.shape[0])
    print("Grouped usable pairs:", grouped.shape[0])
    print("Unique antibiotics with usable labels:", stats.shape[0])

    print("\nTop 20 antibiotics by labeled count:")
    for ab, row in stats.head(20).iterrows():
        print(
            f"  {ab:30s} n={int(row['count'])} R={int(row['sum'])} S={int(row['neg'])}")

    # ----------------------------
    # Filter antibiotics to trainable ones
    # ----------------------------
    keep_antibiotics = stats[
        (stats["count"] >= args.min_samples) &
        (stats["sum"] >= args.min_pos) &
        (stats["neg"] >= args.min_neg)
    ].index.tolist()

    if len(keep_antibiotics) == 0:
        print("\nNo antibiotics passed thresholds:")
        print(
            f"  min_samples={args.min_samples}, min_pos={args.min_pos}, min_neg={args.min_neg}")
        print("Try for example: --min_samples 50 --min_pos 10 --min_neg 10")
        return

    # ----------------------------
    # Build X_index + Y (rows=genomes with >=1 kept label)
    # ----------------------------
    filtered = grouped[grouped["Antibiotic"].isin(keep_antibiotics)].copy()
    ab_to_col = {ab: j for j, ab in enumerate(keep_antibiotics)}

    genome_label_map = defaultdict(dict)  # kmer_index -> {col: label}
    genome_biosample_map = {}            # kmer_index -> BioSample

    for _, row in filtered.iterrows():
        idx = int(row["kmer_index"])
        col = ab_to_col[row["Antibiotic"]]
        lab = int(row["label"])
        genome_label_map[idx][col] = lab
        genome_biosample_map[idx] = row["BioSample"]

    kept_indices = sorted(genome_label_map.keys())

    Y = np.full((len(kept_indices), len(keep_antibiotics)), -1, dtype=np.int8)
    biosamples = []

    for r, idx in enumerate(kept_indices):
        biosamples.append(genome_biosample_map.get(idx, ""))
        for col, lab in genome_label_map[idx].items():
            Y[r, col] = lab

    X_index = np.array(kept_indices, dtype=np.int32)

    # ----------------------------
    # Save outputs
    # ----------------------------
    ab_path = os.path.join(args.out_dir, "antibiotics.txt")
    with open(ab_path, "w") as f:
        for ab in keep_antibiotics:
            f.write(ab + "\n")

    np.save(os.path.join(args.out_dir, "X_index.npy"), X_index)
    np.save(os.path.join(args.out_dir, "Y.npy"), Y)
    np.save(os.path.join(args.out_dir, "biosamples.npy"),
            np.array(biosamples, dtype=object))

    stats_out = stats.reset_index().rename(columns={"index": "Antibiotic"})
    stats_out.to_csv(os.path.join(
        args.out_dir, "antibiotic_label_stats.csv"), index=False)

    # ----------------------------
    # Print summary
    # ----------------------------
    print("\n=== build_antibiotic_labels.py DONE ===")
    print(f"Total phenotype rows (merged): {merged.shape[0]}")
    print(f"Usable labeled (BioSample, Antibiotic) pairs: {grouped.shape[0]}")
    print(f"Antibiotics kept: {len(keep_antibiotics)}")
    print(f"Saved: {ab_path}")
    print(f"Saved: {args.out_dir}/X_index.npy  (len={len(X_index)})")
    print(
        f"Saved: {args.out_dir}/Y.npy        (shape={Y.shape}, -1=missing label)")
    print(f"Saved: {args.out_dir}/biosamples.npy")
    print(f"Saved: {args.out_dir}/antibiotic_label_stats.csv")

    print("\nTop kept antibiotics by sample count:")
    kept_stats = stats.loc[keep_antibiotics].sort_values(
        "count", ascending=False).head(10)
    for ab, row in kept_stats.iterrows():
        print(
            f"  {ab:30s}  n={int(row['count'])}  R={int(row['sum'])}  S={int(row['neg'])}")


if __name__ == "__main__":
    main()
