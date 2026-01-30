#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix

# ----------------------------
# Load data
# ----------------------------
X_full = np.load("data/genomes/kmer_matrix.npy", mmap_mode="r")
X_index = np.load("data/ml/X_index.npy")
Y = np.load("data/ml/Y.npy")
antibiotics = [l.strip() for l in open("data/ml/antibiotics.txt") if l.strip()]

X = X_full[X_index]

results = []

# ----------------------------
# Train per antibiotic
# ----------------------------
for j, ab in enumerate(antibiotics):
    y = Y[:, j]
    mask = y != -1

    Xj = X[mask]
    yj = y[mask]

    # skip if too small (safety)
    if len(np.unique(yj)) < 2 or len(yj) < 30:
        continue

    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42)

    aucs = []
    sens = []
    spec = []

    for train_idx, test_idx in splitter.split(Xj, yj):
        X_train, X_test = Xj[train_idx], Xj[test_idx]
        y_train, y_test = yj[train_idx], yj[test_idx]

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test, probs))

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))

    results.append({
        "antibiotic": ab,
        "n": len(yj),
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "sensitivity": np.mean(sens),
        "specificity": np.mean(spec),
    })

# ----------------------------
# Save results
# ----------------------------
df = pd.DataFrame(results).sort_values("auc_mean", ascending=False)
df.to_csv("data/ml/baseline_results.csv", index=False)

print("\n=== BASELINE RESULTS ===")
print(df.to_string(index=False))
print("\nSaved: data/ml/baseline_results.csv")
