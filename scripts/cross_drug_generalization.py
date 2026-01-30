#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

TASK_DIR="ml_tasks"
LAT_DIR="latents/ae256"
OUT_DIR="results/crossdrug_v1"
OUT=os.path.join(OUT_DIR,"crossdrug_results.csv")
SEED=1337
ALPHA=0.003

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    Ltr=np.load(os.path.join(LAT_DIR,"L_train.npy"), mmap_mode="r")
    Lte=np.load(os.path.join(LAT_DIR,"L_test.npy"),  mmap_mode="r")

    drugs=sorted([f[:-5] for f in os.listdir(TASK_DIR) if f.endswith(".json")])

    rows=[]
    for heldout in drugs:
        Xs=[]; ys=[]

        # pool all OTHER drugs for training
        for d in drugs:
            if d==heldout:
                continue
            req = [f"{d}_train_idx.npy", f"{d}_train_y.npy"]
            if not all(os.path.exists(os.path.join(TASK_DIR,r)) for r in req):
                continue

            y=np.load(os.path.join(TASK_DIR,f"{d}_train_y.npy")).astype(int)
            if len(np.unique(y))<2:
                continue
            idx=np.load(os.path.join(TASK_DIR,f"{d}_train_idx.npy"))
            Xs.append(Ltr[idx])
            ys.append(y)

        if not Xs:
            continue
        X=np.vstack(Xs)
        y=np.concatenate(ys)

        clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=ALPHA,
                random_state=SEED,
                max_iter=3000,
                tol=1e-3
            )
        )
        clf.fit(X,y)

        # heldout test
        req = [f"{heldout}_test_idx.npy", f"{heldout}_test_y.npy"]
        if not all(os.path.exists(os.path.join(TASK_DIR,r)) for r in req):
            continue
        te_y=np.load(os.path.join(TASK_DIR,f"{heldout}_test_y.npy")).astype(int)
        if len(np.unique(te_y))<2:
            continue
        te_idx=np.load(os.path.join(TASK_DIR,f"{heldout}_test_idx.npy"))
        p=clf.predict_proba(Lte[te_idx])[:,1]

        auprc=float(average_precision_score(te_y,p))
        prev=float(te_y.mean())

        rows.append({
            "heldout_drug": heldout,
            "auprc": auprc,
            "auroc": float(roc_auc_score(te_y,p)),
            "test_n": int(len(te_y)),
            "test_prev": prev,
            "auprc_over_prev": float(auprc/(prev+1e-12)),
        })

    df=pd.DataFrame(rows).sort_values("auprc", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"✓ wrote {OUT}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
