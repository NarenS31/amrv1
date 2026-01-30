#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

TASK_DIR="ml_tasks"
LAT_DIR="latents/ae256"
MODEL_DIR="results/stronger_v1"
OUT_DIR="results/mechanisms_v1/xgb_contribs"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        import xgboost as xgb
    except Exception as e:
        raise SystemExit(f"❌ xgboost import failed: {e}")

    Lte=np.load(os.path.join(LAT_DIR,"L_test.npy"), mmap_mode="r")

    drugs=sorted([f[:-5] for f in os.listdir(TASK_DIR) if f.endswith(".json")])

    summary=[]
    for drug in drugs:
        model_path=os.path.join(MODEL_DIR, f"{drug}_xgb.json")
        te_idx_path=os.path.join(TASK_DIR, f"{drug}_test_idx.npy")
        te_y_path=os.path.join(TASK_DIR, f"{drug}_test_y.npy")
        if not (os.path.exists(model_path) and os.path.exists(te_idx_path) and os.path.exists(te_y_path)):
            continue

        te_idx=np.load(te_idx_path)
        te_y=np.load(te_y_path).astype(int)
        if len(np.unique(te_y)) < 2:
            continue

        booster=xgb.Booster()
        booster.load_model(model_path)

        X=Lte[te_idx]
        dm=xgb.DMatrix(X)

        # contributions: [n, d+1] last column is bias term
        contrib = booster.predict(dm, pred_contribs=True)
        contrib = np.array(contrib)
        d = contrib.shape[1] - 1

        mean_abs = np.mean(np.abs(contrib[:, :d]), axis=0)
        top = np.argsort(mean_abs)[::-1][:25]

        out = pd.DataFrame({
            "drug": drug,
            "latent_dim": top,
            "mean_abs_contrib": mean_abs[top],
        })
        out_csv=os.path.join(OUT_DIR, f"{drug}_top_latent_dims.csv")
        out.to_csv(out_csv, index=False)

        summary.append({
            "drug": drug,
            "top_dim_1": int(top[0]),
            "top_dim_2": int(top[1]) if len(top) > 1 else None,
            "top_dim_3": int(top[2]) if len(top) > 2 else None,
            "top_dim_1_mean_abs": float(mean_abs[top[0]]),
        })

        print("✓ wrote", out_csv)

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join("results/mechanisms_v1","xgb_contribs_summary.csv"), index=False)
        print("✓ wrote results/mechanisms_v1/xgb_contribs_summary.csv")
    else:
        print("No XGB contrib outputs (no models found?).")

if __name__ == "__main__":
    main()
