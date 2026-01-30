#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

MASTER="results/master_v1/master_results.csv"
OUT_DIR="results/diffusion_utility_v1"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m = pd.read_csv(MASTER)

    # Only rows where diffusion exists
    m = m[m["diff_test_auprc"].notna()].copy()
    if m.empty:
        raise SystemExit("❌ No diffusion rows found in master_results.csv")

    m["diff_gain"] = m["diff_test_auprc"] - m["lr_test_auprc"]

    keep = m[["drug","test_n_pos","test_n","test_prevalence","lr_test_auprc","diff_test_auprc","diff_gain"]].sort_values("diff_gain", ascending=False)
    out_csv = os.path.join(OUT_DIR, "diffusion_gain_table.csv")
    keep.to_csv(out_csv, index=False)
    print("✓ wrote", out_csv)
    print(keep.to_string(index=False))

    # Scatter: gain vs number of positives
    plt.figure()
    plt.scatter(keep["test_n_pos"], keep["diff_gain"])
    plt.xlabel("Test positives (test_n_pos)")
    plt.ylabel("Diffusion gain (AUPRC_aug - AUPRC_base)")
    plt.title("Diffusion utility vs available signal (positives)")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "diffusion_gain_vs_pos.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("✓ wrote", out_png)

if __name__ == "__main__":
    main()
