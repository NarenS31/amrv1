#!/usr/bin/env python3
import os, glob, importlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

AMR_TSV_DIR = "results/amrfinder_test"
IDS_TEST    = "kmers/k31/ids_test.txt"
TASK_DIR    = "ml_tasks"

ML_LR_LATENT = "results/latents_v1/latent_logreg_results.csv"
ML_STRONGER  = "results/stronger_v1/stronger_latent_results.csv"
DIFF_SUMMARY = "results/diffusion_latents_v1/diffusion_aug_results.csv"

OUT_DIR      = "results/amrfinder"
OUT_FAIR_VS  = os.path.join(OUT_DIR, "fair_vs_ml.csv")
OUT_METRICS  = os.path.join(OUT_DIR, "fair_metrics.csv")
OUT_MISSING  = os.path.join(OUT_DIR, "missing_amrfinder_tsvs.txt")

def load_predictor():
    m = importlib.import_module("src.amrfinder_to_prediction")
    if hasattr(m, "predict_amrfinder_rs") and callable(getattr(m, "predict_amrfinder_rs")):
        return getattr(m, "predict_amrfinder_rs"), "predict_amrfinder_rs"
    raise RuntimeError("Need predict_amrfinder_rs(df, drug) in src/amrfinder_to_prediction.py")

def find_tsv_for_genome(genome_id: str) -> str | None:
    p = os.path.join(AMR_TSV_DIR, f"{genome_id}.tsv")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(AMR_TSV_DIR, f"*{genome_id}*.tsv"))
    return hits[0] if hits else None

def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))

def safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))

def load_stronger_any_schema(path):
    df = pd.read_csv(path)
    # accept either old schema or your current one
    cols = df.columns.tolist()

    # possible names observed in your prints:
    # drug, lr, svm_calibrated, xgboost, diffusion, ...
    out = df[["drug"]].copy()

    if "svm_calibrated" in cols:
        out["svm_cal_auprc"] = df["svm_calibrated"]
    elif "svm_calibrated_test_auprc" in cols:
        out["svm_cal_auprc"] = df["svm_calibrated_test_auprc"]

    if "xgboost" in cols:
        out["xgb_auprc"] = df["xgboost"]
    elif "xgb_test_auprc" in cols:
        out["xgb_auprc"] = df["xgb_test_auprc"]

    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    predictor, pname = load_predictor()
    print("[OK] Using predictor:", pname)

    ids_test = [l.strip() for l in open(IDS_TEST) if l.strip()]

    # Pick drugs from stronger results if available; else from tasks on disk
    if os.path.exists(ML_STRONGER):
        drugs = pd.read_csv(ML_STRONGER)["drug"].tolist()
    else:
        drugs = sorted(set([f.replace("_train_y.npy","") for f in os.listdir(TASK_DIR) if f.endswith("_train_y.npy")]))

    missing = []
    rows = []

    for drug in drugs:
        y_path = os.path.join(TASK_DIR, f"{drug}_test_y.npy")
        idx_path = os.path.join(TASK_DIR, f"{drug}_test_idx.npy")
        if not (os.path.exists(y_path) and os.path.exists(idx_path)):
            continue

        y = np.load(y_path).astype(int)
        idx = np.load(idx_path).astype(int)
        genome_ids = [ids_test[i] for i in idx]

        amr_pred = np.zeros_like(y, dtype=int)
        n_missing = 0

        for j, gid in enumerate(genome_ids):
            tsv = find_tsv_for_genome(gid)
            if tsv is None:
                n_missing += 1
                missing.append(gid)
                continue
            try:
                df = pd.read_csv(tsv, sep="\t")
            except Exception:
                n_missing += 1
                missing.append(gid)
                continue

            pred = predictor(df, drug)  # "R"/"S"
            amr_pred[j] = 1 if str(pred).upper().startswith("R") else 0

        auprc = safe_auprc(y, amr_pred)
        auroc = safe_auroc(y, amr_pred)

        rows.append({
            "drug": drug,
            "amrfinder_auprc": auprc,
            "amrfinder_auroc": auroc,
            "test_n": int(len(y)),
            "test_prev": float(y.mean()),
            "amrfinder_pred_mean": float(amr_pred.mean()),
            "missing_tsvs_in_task": int(n_missing),
        })
        print(f"[DONE] {drug:28s} AUPRC={auprc:.4f} AUROC={auroc:.4f} pred_mean={amr_pred.mean():.3f} missing={n_missing}")

    fair = pd.DataFrame(rows).sort_values("drug")
    fair.to_csv(OUT_FAIR_VS, index=False)

    out = fair.copy()

    if os.path.exists(ML_LR_LATENT):
        lr = pd.read_csv(ML_LR_LATENT)[["drug","test_auprc","test_auroc"]].rename(
            columns={"test_auprc":"lr_latent_auprc","test_auroc":"lr_latent_auroc"}
        )
        out = out.merge(lr, on="drug", how="left")

    if os.path.exists(ML_STRONGER):
        st = load_stronger_any_schema(ML_STRONGER)
        out = out.merge(st, on="drug", how="left")

    if os.path.exists(DIFF_SUMMARY):
        df = pd.read_csv(DIFF_SUMMARY)[["drug","test_auprc_aug"]].rename(columns={"test_auprc_aug":"diff_auprc"})
        out = out.merge(df, on="drug", how="left")

    out.to_csv(OUT_METRICS, index=False)

    missing = sorted(set(missing))
    with open(OUT_MISSING, "w") as f:
        for g in missing:
            f.write(g + "\n")

    print("\n✓ wrote", OUT_FAIR_VS)
    print("✓ wrote", OUT_METRICS)
    print("✓ missing TSV list:", OUT_MISSING, f"(n={len(missing)})")

if __name__ == "__main__":
    main()
