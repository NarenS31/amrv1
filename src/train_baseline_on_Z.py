#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix


Z = np.load("data/ml/Z.npy")
Y = np.load("data/ml/Y.npy")
antibiotics = [l.strip() for l in open("data/ml/antibiotics.txt") if l.strip()]

results = []

for j, ab in enumerate(antibiotics):
    y = Y[:, j]
    mask = y != -1
    Zj = Z[mask]
    yj = y[mask]

    if len(np.unique(yj)) < 2 or len(yj) < 30:
        continue

    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42)

    aucs, sens, spec = [], [], []

    for train_idx, test_idx in splitter.split(Zj, yj):
        Z_train, Z_test = Zj[train_idx], Zj[test_idx]
        y_train, y_test = yj[train_idx], yj[test_idx]

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Z_train, y_train)

        probs = clf.predict_proba(Z_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test, probs))
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        sens.append(tp / (tp + fn))
        spec.append(tn / (tn + fp))

    results.append({
        "antibiotic": ab,
        "n": int(len(yj)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "sensitivity": float(np.mean(sens)),
        "specificity": float(np.mean(spec)),
    })

df = pd.DataFrame(results).sort_values("auc_mean", ascending=False)
df.to_csv("data/ml/baseline_results_ae.csv", index=False)

print("\n=== AE-EMBEDDING LOGREG RESULTS ===")
print(df.to_string(index=False))
print("\nSaved: data/ml/baseline_results_ae.csv")
