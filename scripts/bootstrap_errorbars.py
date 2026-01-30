import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pathlib import Path

N_BOOT = 1000
SEED = 1337

def bootstrap(y, p):
    rng = np.random.default_rng(SEED)
    n = len(y)
    scores = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(average_precision_score(y[idx], p[idx]))
    return np.percentile(scores, [2.5, 50, 97.5])

# example: XGBoost
rows = []
res = pd.read_csv("results/stronger_v1/stronger_latent_results.csv")

for drug in res["drug"].unique():
    y = np.load(f"ml_tasks/{drug}_test_y.npy")
    p = np.load(f"results/stronger_v1/{drug}_xgb_test_prob.npy")

    lo, mid, hi = bootstrap(y, p)
    rows.append({
        "drug": drug,
        "model": "xgboost",
        "auprc_median": mid,
        "ci_low": lo,
        "ci_high": hi
    })

pd.DataFrame(rows).to_csv("results/stronger_v1/errorbars_xgb.csv", index=False)
print("✓ saved error bars")
