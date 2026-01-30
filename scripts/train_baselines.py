#!/usr/bin/env python3
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

KMER_DIR = "kmers/k31"
TASK_DIR = "ml_tasks"
OUT_DIR  = "results/baselines"

def load_X(split):
    return np.load(os.path.join(KMER_DIR, f"X_{split}.npy"), mmap_mode="r")

def load_task(path):
    with open(path) as f:
        return json.load(f)

def subset(X, idx):
    # idx is int32 vector, X is memmap
    return X[idx]

def eval_split(model, X, y):
    # probability of class 1
    p = model.predict_proba(X)[:,1]
    auroc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    auprc = average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return auroc, auprc

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train = load_X("train")
    X_val   = load_X("val")
    X_test  = load_X("test")

    rows = []
    task_files = sorted(glob.glob(os.path.join(TASK_DIR, "*.json")))
    if not task_files:
        raise SystemExit("No tasks found in ml_tasks/*.json")

    total = len(task_files)
    for i, tf in enumerate(task_files, 1):
        print(f"\n[{i}/{total}] training {os.path.basename(tf).replace('.json','')}", flush=True)

        task = load_task(tf)
        drug = task["drug"]

        idx_tr = np.load(os.path.join(TASK_DIR, task["splits"]["train"]["idx_file"]))
        y_tr   = np.load(os.path.join(TASK_DIR, task["splits"]["train"]["y_file"]))

        idx_va = np.load(os.path.join(TASK_DIR, task["splits"]["val"]["idx_file"]))
        y_va   = np.load(os.path.join(TASK_DIR, task["splits"]["val"]["y_file"]))

        idx_te = np.load(os.path.join(TASK_DIR, task["splits"]["test"]["idx_file"]))
        y_te   = np.load(os.path.join(TASK_DIR, task["splits"]["test"]["y_file"]))

        # subset matrices
        Xtr = subset(X_train, idx_tr)
        Xva = subset(X_val, idx_va)
        Xte = subset(X_test, idx_te)

        # class weights for imbalance
        model = SGDClassifier(
            loss="log_loss",
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            tol=1e-3
        )
        model.fit(Xtr, y_tr)

        va_auroc, va_auprc = eval_split(model, Xva, y_va)
        te_auroc, te_auprc = eval_split(model, Xte, y_te)

        rows.append({
            "drug": drug,
            "train_n": len(y_tr),
            "train_pos": int(y_tr.sum()),
            "train_neg": int(len(y_tr)-y_tr.sum()),
            "val_auroc": va_auroc,
            "val_auprc": va_auprc,
            "test_auroc": te_auroc,
            "test_auprc": te_auprc,
        })

        # save per-drug model coefficients (optional, for interpretability)
        np.save(os.path.join(OUT_DIR, f"{drug}_coef.npy"), model.coef_.astype(np.float32))
        np.save(os.path.join(OUT_DIR, f"{drug}_intercept.npy"), model.intercept_.astype(np.float32))

        print(f"{drug:30s}  test AUROC={te_auroc:.3f}  AUPRC={te_auprc:.3f}")

    df = pd.DataFrame(rows).sort_values("test_auprc", ascending=False)
    out_csv = os.path.join(OUT_DIR, "logreg_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n✓ Wrote:", out_csv)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
