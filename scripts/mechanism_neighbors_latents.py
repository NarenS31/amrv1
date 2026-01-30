#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

TASK_DIR="ml_tasks"
LAT_DIR="latents/ae256"
PROB_DIR="results/stronger_v1"
IDS_TRAIN="kmers/k31/ids_train.txt"
IDS_TEST ="kmers/k31/ids_test.txt"

OUT_DIR="results/mechanisms_v1/neighbors"
TOPQ=10       # top/bottom test queries per drug
TOPN=10       # nearest neighbors per query

def load_ids(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    Ltr=np.load(os.path.join(LAT_DIR,"L_train.npy"), mmap_mode="r")
    Lte=np.load(os.path.join(LAT_DIR,"L_test.npy"),  mmap_mode="r")

    ids_tr = load_ids(IDS_TRAIN)
    ids_te = load_ids(IDS_TEST)

    drugs=sorted([f[:-5] for f in os.listdir(TASK_DIR) if f.endswith(".json")])

    all_rows=[]
    for drug in drugs:
        tr_idx_path=os.path.join(TASK_DIR,f"{drug}_train_idx.npy")
        te_idx_path=os.path.join(TASK_DIR,f"{drug}_test_idx.npy")
        tr_y_path=os.path.join(TASK_DIR,f"{drug}_train_y.npy")
        te_y_path=os.path.join(TASK_DIR,f"{drug}_test_y.npy")

        if not (os.path.exists(tr_idx_path) and os.path.exists(te_idx_path) and os.path.exists(tr_y_path) and os.path.exists(te_y_path)):
            continue

        tr_idx=np.load(tr_idx_path)
        te_idx=np.load(te_idx_path)
        tr_y=np.load(tr_y_path).astype(int)
        te_y=np.load(te_y_path).astype(int)

        if len(np.unique(tr_y))<2 or len(np.unique(te_y))<2:
            continue

        prob_path=os.path.join(PROB_DIR,f"{drug}_xgb_test_prob.npy")
        if not os.path.exists(prob_path):
            prob_path=os.path.join(PROB_DIR,f"{drug}_svm_test_prob.npy")
        if not os.path.exists(prob_path):
            continue
        p=np.load(prob_path)
        if p.shape[0] != te_idx.shape[0]:
            # mismatch means wrong file alignment; skip loudly
            print(f"⚠️ {drug}: prob length {len(p)} != test idx length {len(te_idx)}; skipping")
            continue

        # pick query genomes: highest predicted R and lowest predicted R
        order=np.argsort(p)
#         low_q=order[:TOPQ]
#         high_q=order[-TOPQ:][::-1]

        Xtr=Ltr[tr_idx]
        ytr=tr_y

        rows=[]
        for bucket, qset in [("all_test", order)]:
            for qrank, qpos in enumerate(qset, 1):
                q_lat = Lte[te_idx[qpos]].reshape(1,-1)
                # cosine distance to all train
                d = cosine_distances(q_lat, Xtr).ravel()
                nn = np.argsort(d)[:TOPN]

                q_gid = ids_te[te_idx[qpos]]
                q_true = int(te_y[qpos])
                q_prob = float(p[qpos])

                # ---- append canonical per-query prediction (once per query) ----
                os.makedirs('results/eval', exist_ok=True)
                outp = 'results/eval/test_predictions_full.csv'
                header_needed = not os.path.exists(outp)
                with open(outp, 'a') as f:
                    if header_needed:
                        f.write('genome_id,drug,y_true,y_score\n')
                    f.write(f"{q_gid},{drug},{int(q_true)},{float(q_prob)}\n")
                for nrank, j in enumerate(nn, 1):
                    tr_global = int(tr_idx[j])
                    rows.append({
                        "drug": drug,
                        "bucket": bucket,
                        "query_rank": qrank,
                        "query_genome": q_gid,
                        "query_true_y": q_true,
                        "query_prob_R": q_prob,
                        "nn_rank": nrank,
                        "nn_genome": ids_tr[tr_global],
                        "nn_true_y": int(ytr[j]),
                        "cosine_dist": float(d[j]),
                    })

        out_csv=os.path.join(OUT_DIR,f"{drug}_neighbors.csv")
        df=pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print("✓ wrote", out_csv, f"({len(df)} rows)")
        all_rows.append(df)

    if all_rows:
        big=pd.concat(all_rows, ignore_index=True)
        big.to_csv(os.path.join("results/mechanisms_v1","neighbors_all_drugs.csv"), index=False)
        print("✓ wrote results/mechanisms_v1/neighbors_all_drugs.csv")
    else:
        print("No neighbor files produced (check ids paths / prob files).")
if __name__ == "__main__":
    main()
