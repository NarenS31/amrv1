#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

PROB_DIR = "results/stronger_v1"
TASK_DIR = "ml_tasks"
OUT_DIR  = "results/calibration_v1"
N_BINS = 10

def ece(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    out = 0.0
    n = len(y)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        out += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(out)

def brier(y, p):
    return float(np.mean((p - y) ** 2))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    drugs = sorted(set(
        f.replace("_svm_val_prob.npy","")
        for f in os.listdir(PROB_DIR)
        if f.endswith("_svm_val_prob.npy")
    ))

    rows = []
    for drug in drugs:
        y_path = os.path.join(TASK_DIR, f"{drug}_val_y.npy")
        if not os.path.exists(y_path):
            continue
        y = np.load(y_path).astype(int)
        if len(np.unique(y)) < 2:
            continue

        svm_p = np.load(os.path.join(PROB_DIR, f"{drug}_svm_val_prob.npy")).astype(float)
        xgb_p = np.load(os.path.join(PROB_DIR, f"{drug}_xgb_val_prob.npy")).astype(float)

        for model, p in [("svm_platt", svm_p), ("xgboost", xgb_p)]:
            rows.append({
                "drug": drug,
                "split": "val",
                "model": model,
                "n": int(len(y)),
                "prevalence": float(y.mean()),
                "ece": ece(y, p, N_BINS),
                "brier": brier(y, p),
            })

        frac_svm, mean_svm = calibration_curve(y, svm_p, n_bins=N_BINS, strategy="uniform")
        frac_xgb, mean_xgb = calibration_curve(y, xgb_p, n_bins=N_BINS, strategy="uniform")

        plt.figure()
        plt.plot(mean_svm, frac_svm, marker="o", label="SVM (Platt)")
        plt.plot(mean_xgb, frac_xgb, marker="o", label="XGBoost")
        plt.plot([0,1],[0,1], "--", label="Perfect")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title(f"Calibration (VAL) — {drug}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{drug}_val_calibration.png"), dpi=300)
        plt.close()

    df = pd.DataFrame(rows).sort_values(["drug","model"])
    out_csv = os.path.join(OUT_DIR, "calibration_metrics.csv")
    df.to_csv(out_csv, index=False)

    print("✓ saved:", out_csv)
    print("✓ plots :", plot_dir)

if __name__ == "__main__":
    main()
