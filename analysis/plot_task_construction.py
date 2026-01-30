import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("results/stronger_v1/stronger_latent_results.csv")
xgb = df[df["model"] == "xgboost"].sort_values("test_prev")

plt.figure(figsize=(9,6))

# scatter: prevalence vs AUPRC
plt.scatter(
    xgb["test_prev"],
    xgb["test_auprc"],
    s=xgb["test_n"] * 2,   # size = test set size
    alpha=0.8
)

# label hard tasks only
for _, r in xgb.iterrows():
    if r["test_prev"] < 0.4:
        plt.annotate(r["drug"], (r["test_prev"], r["test_auprc"]), fontsize=9)

plt.xlabel("Test prevalence (fraction resistant)")
plt.ylabel("AUPRC")
plt.title("Task Construction per Antibiotic\n(size = test set size, imbalance drives apparent performance)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/task_construction_per_antibiotic.png", dpi=300)
plt.show()
