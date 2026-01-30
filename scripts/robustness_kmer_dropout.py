#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

Z_DIR = "kmers_proj/rp8192"
TASK_DIR = "ml_tasks"
OUT_DIR = "results/robustness_v1"
OUT_CSV = os.path.join(OUT_DIR, "kmer_dropout_results.csv")

SEED = 1337
ALPHA = 0.003

# Pick 2 drugs: one "easy/high-prev" and one "hard/low-prev"
DRUGS = ["ciprofloxacin", "amikacin"]

# dropout rates = fraction of features set to zero (simulate missing k-mer evidence)
DROPOUTS = [0.0, 0.05, 0.10, 0.20, 0.30]

def apply_feature_dropout(X: np.ndarray, drop: float, rng: np.random.Generator) -> np.ndarray:
    if drop <= 0:
        return X
    mask = rng.random(X.shape) > drop
    return X * mask

def load_task(drug: str, split: str):
    idx = np.load(os.path.join(TASK_DIR, f"{drug}_{split}_idx.npy"))
    y = np.load(os.path.join(TASK_DIR, f"{drug}_{split}_y.npy")).astype(int)
    return idx, y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    Ztr = np.load(os.path.join(Z_DIR, "Z_train.npy"), mmap_mode="r")
    Zte = np.load(os.path.join(Z_DIR, "Z_test.npy"),  mmap_mode="r")

    rows = []
    for drug in DRUGS:
        tr_idx, tr_y = load_task(drug, "train")
        te_idx, te_y = load_task(drug, "test")

        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            print(f"Skipping {drug}: not enough class diversity")
            continue

        # Train once on clean train data (represents training on complete/normal assemblies)
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=ALPHA,
                random_state=SEED,
                max_iter=5000,
                tol=1e-3
            )
        )
        clf.fit(Ztr[tr_idx], tr_y)

        base_prev = float(te_y.mean())
        for drop in DROPOUTS:
            rng = np.random.default_rng(SEED + int(drop * 1000))

            X = Zte[te_idx]
            Xc = apply_feature_dropout(X, drop, rng)

            p = clf.predict_proba(Xc)[:, 1]
            auprc = float(average_precision_score(te_y, p))
            auroc = float(roc_auc_score(te_y, p))

            rows.append({
                "drug": drug,
                "dropout": drop,
                "test_n": int(len(te_y)),
                "test_prev": base_prev,
                "auprc": auprc,
                "auroc": auroc,
                "auprc_over_prev": float(auprc / (base_prev + 1e-12)),
            })

    df = pd.DataFrame(rows).sort_values(["drug", "dropout"])
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ wrote {OUT_CSV}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
