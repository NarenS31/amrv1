#!/usr/bin/env python3
import argparse, glob, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

def load_npy(path):
    arr = np.load(path)
    return arr.squeeze()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob_glob", default="results/stronger_v1/*_prob.npy",
                    help="Glob for per-drug probability files.")
    ap.add_argument("--label_glob", default="results/stronger_v1/*_y_true.npy",
                    help="Glob for per-drug y_true files (0/1). If you don't have these, you must generate them.")
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="results/final_tables/bootstrap_cis.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prob_files = sorted(glob.glob(args.prob_glob))
    y_files = sorted(glob.glob(args.label_glob))

    if not prob_files:
        raise SystemExit(f"No prob files matched: {args.prob_glob}")
    if not y_files:
        raise SystemExit(f"No y_true files matched: {args.label_glob}\nYou need saved test labels per drug (0/1).")

    # map drug+model from filenames like: drug__MODEL_prob.npy OR drug_MODEL_prob.npy
    def key(p):
        b = os.path.basename(p)
        b = b.replace("_prob.npy","").replace(".npy","")
        return b

    probs = {key(p): p for p in prob_files}
    ys = {key(p).replace("_y_true",""): p for p in y_files}  # support *_y_true.npy

    rows = []
    for k, ppath in probs.items():
        # try to find matching y
        ypath = None
        if k in ys: ypath = ys[k]
        if ypath is None and (k + "_y_true") in ys: ypath = ys[k + "_y_true"]
        if ypath is None:
            # last resort: strip model suffix after __
            base = k.split("__")[0]
            if base in ys: ypath = ys[base]
        if ypath is None:
            print("skip (no y_true):", k)
            continue

        y = load_npy(ypath).astype(int)
        p = load_npy(ppath).astype(float)
        if len(y) != len(p):
            print("skip (len mismatch):", k, len(y), len(p))
            continue
        if len(np.unique(y)) < 2:
            print("skip (one class):", k)
            continue

        # point estimate
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)

        idx = np.arange(len(y))
        b_auroc = []
        b_auprc = []
        for _ in range(args.n_boot):
            samp = rng.choice(idx, size=len(idx), replace=True)
            ysamp = y[samp]
            psamp = p[samp]
            if len(np.unique(ysamp)) < 2:
                continue
            b_auroc.append(roc_auc_score(ysamp, psamp))
            b_auprc.append(average_precision_score(ysamp, psamp))

        b_auroc = np.array(b_auroc)
        b_auprc = np.array(b_auprc)

        def ci(a):
            return np.nanpercentile(a, [2.5, 50, 97.5])

        auroc_ci = ci(b_auroc)
        auprc_ci = ci(b_auprc)

        rows.append({
            "task": k,
            "n": len(y),
            "prev": float(y.mean()),
            "auroc": auroc,
            "auroc_lo": auroc_ci[0],
            "auroc_med": auroc_ci[1],
            "auroc_hi": auroc_ci[2],
            "auprc": auprc,
            "auprc_lo": auprc_ci[0],
            "auprc_med": auprc_ci[1],
            "auprc_hi": auprc_ci[2],
        })

    out = pd.DataFrame(rows).sort_values(["auprc"], ascending=False)
    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("✓ wrote", args.out_csv)
    print("rows:", len(out))

if __name__ == "__main__":
    main()
