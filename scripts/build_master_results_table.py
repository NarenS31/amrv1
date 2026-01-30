#!/usr/bin/env python3
import os
import pandas as pd

LR   = "results/latents_v1/latent_logreg_results.csv"
DIFF = "results/diffusion_latents_v1/diffusion_aug_results.csv"
STR  = "results/stronger_v1/stronger_latent_results.csv"
CAL  = "results/calibration_v1/calibration_metrics.csv"

OUT_DIR = "results/master_v1"
OUT = os.path.join(OUT_DIR, "master_results.csv")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- LR baseline ---
    lr = pd.read_csv(LR)
    lr = lr[[
        "drug","test_auprc","test_auroc","test_prevalence",
        "test_n","test_n_pos","test_n_neg"
    ]].rename(columns={
        "test_auprc":"lr_test_auprc",
        "test_auroc":"lr_test_auroc",
    })
    lr = lr.drop_duplicates(subset=["drug"], keep="first")

    # --- diffusion aug ---
    diff = pd.read_csv(DIFF)
    diff = diff[["drug","added_synth","val_auprc_aug","test_auprc_aug"]].rename(columns={
        "val_auprc_aug":"diff_val_auprc",
        "test_auprc_aug":"diff_test_auprc",
    })
    diff = diff.drop_duplicates(subset=["drug"], keep="first")

    # --- stronger models (LONG -> WIDE) ---
    strg = pd.read_csv(STR)
    # expects columns: drug, model, test_auprc
    if not set(["drug","model","test_auprc"]).issubset(strg.columns):
        raise SystemExit(f"{STR} missing required cols. Have: {strg.columns.tolist()}")

    strg_w = strg.pivot_table(index="drug", columns="model", values="test_auprc", aggfunc="first").reset_index()

    # normalize model names -> consistent columns
    ren = {}
    if "svm_calibrated" in strg_w.columns:
        ren["svm_calibrated"] = "svm_platt_test_auprc"
    if "xgboost" in strg_w.columns:
        ren["xgboost"] = "xgb_test_auprc"
    strg_w = strg_w.rename(columns=ren)

    # --- calibration (val) wide ---
    cal = pd.read_csv(CAL)
    cal = cal[cal["split"].eq("val")]
    cal_w = cal.pivot_table(
        index="drug",
        columns="model",
        values=["ece","brier","n","prevalence"],
        aggfunc="first"
    )
    cal_w.columns = [f"val_{a}_{b}" for a,b in cal_w.columns]
    cal_w = cal_w.reset_index()

    # --- merge ---
    m = lr.merge(strg_w, on="drug", how="left") \
          .merge(diff, on="drug", how="left") \
          .merge(cal_w, on="drug", how="left")

    # derived "best model"
    candidates = ["lr_test_auprc","diff_test_auprc","svm_platt_test_auprc","xgb_test_auprc"]
    for c in candidates:
        if c not in m.columns:
            m[c] = pd.NA
    m["best_test_auprc"] = m[candidates].astype(float).max(axis=1)
    m["best_model"] = m[candidates].astype(float).idxmax(axis=1)
    m["delta_best_vs_lr"] = m["best_test_auprc"] - m["lr_test_auprc"].astype(float)

    m = m.sort_values("lr_test_auprc", ascending=False)
    m.to_csv(OUT, index=False)

    print(f"✓ wrote {OUT}")
    print(m.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
