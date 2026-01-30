# src_nn/lr_baseline.py
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def autodetect_x_path() -> str:
    candidates = [
        "data/genomes/kmer_matrix.npy",
        "data/genomes/kmer_features.npy",
        "data/genomes/X_train.npy",
        "data/ml/X_index.npy",
        "data/ml/X_reduced.npy",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not autodetect X .npy file. Pass --x_path explicitly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", type=str, default=None)
    ap.add_argument("--y_path", type=str, default="data/ml/Y.npy")
    ap.add_argument("--antibiotics_path", type=str,
                    default="data/ml/antibiotics.txt")
    ap.add_argument("--out_csv", type=str,
                    default="results/lr_per_antibiotic.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    x_path = args.x_path or autodetect_x_path()
    X = np.load(x_path).astype(np.float32, copy=False)
    Y = np.load(args.y_path, allow_pickle=True).astype(np.float32, copy=False)

    n, d = X.shape
    t = Y.shape[1]

    if os.path.exists(args.antibiotics_path):
        ab = np.loadtxt(args.antibiotics_path, dtype=str).tolist()
        if len(ab) != t:
            ab = [f"ab_{i}" for i in range(t)]
    else:
        ab = [f"ab_{i}" for i in range(t)]

    rows = []
    for j in range(t):
        yj = Y[:, j]
        m = yj != -1
        if m.sum() < 20 or len(np.unique(yj[m])) < 2:
            rows.append({"antibiotic": ab[j], "n": int(
                m.sum()), "test_auroc": np.nan})
            continue

        Xj = X[m]
        yj2 = yj[m].astype(int)

        Xtr, Xte, ytr, yte = train_test_split(
            Xj, yj2, test_size=0.2, random_state=args.seed, stratify=yj2)

        clf = LogisticRegression(
            C=args.C, max_iter=2000, class_weight="balanced", n_jobs=-1, solver="liblinear"
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        rows.append({"antibiotic": ab[j], "n": int(
            m.sum()), "test_auroc": float(auc)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.sort_values("test_auroc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
