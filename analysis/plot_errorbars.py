import os
import pandas as pd
import matplotlib.pyplot as plt

in_path = "results/metrics/per_drug_metrics_with_ci.csv"
df = pd.read_csv(in_path)

for metric in ["F1", "AUROC", "AUPRC", "MCC"]:
    if f"{metric}_mean" not in df.columns:
        continue

    # For AUROC/AUPRC only plot drugs with enough positives/negatives
    d = df.copy()
    if metric in ["AUROC", "AUPRC"]:
        d = d[d["auroc_auprc_valid"] == 1]

    if len(d) == 0:
        print(f"Skipping {metric}: no valid rows.")
        continue

    x = d["drug"].astype(str)
    y = d[f"{metric}_mean"]
    yerr_low = y - d[f"{metric}_ci_low"]
    yerr_high = d[f"{metric}_ci_high"] - y

    plt.figure(figsize=(12, 5))
    plt.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="o", capsize=3)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel(metric)
    plt.title(f"Per-drug {metric} with 95% bootstrap CI")
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = f"figures/per_drug_{metric.lower()}_ci.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✓ Saved: {out}")

print("Done.")
