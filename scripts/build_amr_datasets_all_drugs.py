#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd

SPLIT_CSV = "results/mash/genome_cluster_split.csv"
PHENO_CSV = "data/bvbrc/phenotype.csv"
OUT_DIR = "datasets/amr_by_drug"

MIN_SAMPLES = 1000   # skip tiny drugs at first (change to 300 if you want everything)
KEEP_SPLITS = {"train", "val", "test"}

def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def slug(s: str) -> str:
    s = norm(s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def canonical_genome_key(x: str):
    """
    Canonical join key to match BV-BRC phenotype Genome ID to your split genome_id.

    BV-BRC phenotypes use: 573.14261
    Your fasta-derived IDs often look like:
      - accn_573.14261.con.0023
      - 573.14261.con.0023
    This extracts the first digits.digits pattern if present.
    """
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r"^accn_", "", s)
    m = re.search(r"(\d+\.\d+)", s)
    if m:
        return m.group(1)
    return s

def main():
    if not os.path.exists(PHENO_CSV):
        print(f"ERROR: phenotype file not found: {PHENO_CSV}")
        sys.exit(1)

    if not os.path.exists(SPLIT_CSV):
        print(f"ERROR: split file not found: {SPLIT_CSV}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load splits ----
    splits = pd.read_csv(SPLIT_CSV)
    if not {"genome_id", "split"}.issubset(set(splits.columns)):
        print("ERROR: split CSV missing required columns. Got:", splits.columns.tolist())
        sys.exit(2)

    splits = splits[splits["split"].isin(KEEP_SPLITS)].copy()
    splits["genome_id"] = splits["genome_id"].astype(str)
    splits["join_key"] = splits["genome_id"].map(canonical_genome_key)

    # ---- load phenotypes ----
    ph = pd.read_csv(PHENO_CSV, low_memory=False)

    # support either exact BV-BRC headers or slightly different casing
    cols = {c.lower(): c for c in ph.columns}

    def col(name):
        return cols.get(name.lower())

    genome_col = col("Genome ID")
    abx_col = col("Antibiotic")
    pheno_col = col("Resistant Phenotype")
    evid_col = col("Evidence")

    if genome_col is None or abx_col is None or pheno_col is None:
        print("ERROR: phenotype CSV missing required columns.")
        print("Columns:", ph.columns.tolist())
        sys.exit(3)

    keep = [genome_col, abx_col, pheno_col] + ([evid_col] if evid_col else [])
    ph = ph[keep].copy()
    ph.rename(columns={
        genome_col: "join_key",
        abx_col: "antibiotic",
        pheno_col: "phenotype",
        **({evid_col: "evidence"} if evid_col else {})
    }, inplace=True)

    ph["join_key"] = ph["join_key"].astype(str).map(canonical_genome_key)
    ph["antibiotic_norm"] = ph["antibiotic"].map(norm)
    ph["phenotype_norm"] = ph["phenotype"].map(norm)

    # Filter: Resistant/Susceptible only
    ph = ph[ph["phenotype_norm"].isin({"resistant", "susceptible"})].copy()

    # Filter: Laboratory Method only (if evidence column exists)
    if "evidence" in ph.columns:
        ph["evidence_norm"] = ph["evidence"].map(norm)
        ph = ph[ph["evidence_norm"].str.contains("laboratory", na=False)].copy()

    # ---- join ----
    df = ph.merge(splits, on="join_key", how="inner")

    if df.empty:
        print("ERROR: join produced 0 rows.")
        print("Example phenotype join_key:", ph["join_key"].dropna().head(10).tolist())
        print("Example split join_key:", splits["join_key"].dropna().head(10).tolist())
        print("\nFix: your split genome_id doesn't contain BV-BRC numeric IDs. Then we need an ID mapping from fasta paths.")
        sys.exit(4)

    # Encode label
    df["y"] = df["phenotype_norm"].map({"susceptible": 0, "resistant": 1}).astype(int)

    # ---- write per-drug ----
    summary_rows = []

    for abx, d in df.groupby("antibiotic_norm"):
        n = len(d)
        if n < MIN_SAMPLES:
            continue

        out_sub = os.path.join(OUT_DIR, slug(abx))
        os.makedirs(out_sub, exist_ok=True)

        for split in ["train", "val", "test"]:
            ds = d[d["split"] == split].copy()
            ds.to_csv(os.path.join(out_sub, f"{split}.csv"), index=False)

        r = int(d["y"].sum())
        s = n - r
        r_pct = 100.0 * r / n if n else 0.0

        summary_rows.append({
            "antibiotic": abx,
            "n": n,
            "resistant": r,
            "susceptible": s,
            "resistant_pct": round(r_pct, 2),
            "train_n": int((d["split"]=="train").sum()),
            "val_n": int((d["split"]=="val").sum()),
            "test_n": int((d["split"]=="test").sum()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(["n"], ascending=False)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"✓ Phenotype rows after filters (lab + R/S): {len(ph):,}")
    print(f"✓ Joined rows (matched to your split set): {len(df):,}")
    print(f"✓ Wrote per-drug datasets to: {OUT_DIR}")
    print(f"✓ Wrote summary: {summary_path}")

    if not summary.empty:
        print("\nTop 15 drugs by sample count:")
        print(summary.head(15).to_string(index=False))
    else:
        print("\nNo drugs met MIN_SAMPLES =", MIN_SAMPLES)
        print("Lower MIN_SAMPLES if you want smaller drugs too.")

if __name__ == "__main__":
    main()
