#!/usr/bin/env python3
import os
import pandas as pd

INP = "results/baselines/logreg_results.csv"
OUT_DIR = "results/master_v1"
OUT = os.path.join(OUT_DIR, "kmer_lr_baseline.csv")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INP)

    col_drug = None
    col_auprc = None
    col_auroc = None

    for c in df.columns:
        lc = c.lower()
        if col_drug is None and lc in ("drug","antibiotic","antibiotic_norm"):
            col_drug = c
        if col_auprc is None and ("test" in lc and "auprc" in lc):
            col_auprc = c
        if col_auroc is None and ("test" in lc and "auroc" in lc):
            col_auroc = c

    if col_drug is None or col_auprc is None or col_auroc is None:
        raise SystemExit(f"Could not infer columns. Columns={df.columns.tolist()}")

    out = df[[col_drug, col_auprc, col_auroc]].rename(columns={
        col_drug: "drug",
        col_auprc: "kmer_lr_test_auprc",
        col_auroc: "kmer_lr_test_auroc",
    }).drop_duplicates(subset=["drug"], keep="first").sort_values("kmer_lr_test_auprc", ascending=False)

    out.to_csv(OUT, index=False)
    print(f"✓ wrote {OUT}")
    print(out.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
