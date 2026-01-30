import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
)

RNG_SEED = 42
B = 1000                 # bootstrap iterations
MIN_N_TOTAL = 10
MIN_N_CLASS = 3         # min positives AND negatives for AUROC/AUPRC


def safe_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn, fp, fn, tp

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    denom = (tn + fp)
    return (tn / denom) if denom else np.nan

def metric_pack(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    out = {}
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    out["Specificity"] = specificity_score(y_true, y_pred)
    out["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    # AUROC/AUPRC require both classes
    if len(np.unique(y_true)) == 2:
        out["AUROC"] = roc_auc_score(y_true, y_score)
        out["AUPRC"] = average_precision_score(y_true, y_score)
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan
    return out

def ci_from_samples(samples, alpha=0.05):
    lo = np.nanpercentile(samples, 100 * (alpha / 2))
    hi = np.nanpercentile(samples, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    np.random.seed(RNG_SEED)

    parquet_path = "results/eval/test_predictions.parquet"
    csv_path = "results/eval/test_predictions.csv"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Missing results/eval/test_predictions.parquet or .csv\n"
            "Create it with columns: drug, y_true, y_score"
        )

    required = {"drug", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need: {required}")

    df = df.copy()
    df["y_true"] = df["y_true"].astype(int)
    df["y_score"] = df["y_score"].astype(float)

    rows = []

    for drug, d in df.groupby("drug"):
        d = d.dropna(subset=["y_true", "y_score"])
        n = len(d)
        n_pos = int(d["y_true"].sum())
        n_neg = n - n_pos

        if n < MIN_N_TOTAL:
            continue

        y_true = d["y_true"].values
        y_score = d["y_score"].values

        point = metric_pack(y_true, y_score, threshold=0.5)
        metrics_samples = {k: [] for k in point.keys()}

        for _ in range(B):
            idx = np.random.randint(0, n, size=n)
            yt = y_true[idx]
            ys = y_score[idx]
            pack = metric_pack(yt, ys, threshold=0.5)
            for k, v in pack.items():
                metrics_samples[k].append(v)

        out = {
            "drug": drug,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc_auprc_valid": int(n_pos >= MIN_N_CLASS and n_neg >= MIN_N_CLASS),
        }

        for k in point.keys():
            lo, hi = ci_from_samples(np.array(metrics_samples[k]))
            out[f"{k}_mean"] = point[k]
            out[f"{k}_ci_low"] = lo
            out[f"{k}_ci_high"] = hi

        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise SystemExit("No drugs passed filters. Lower MIN_N_TOTAL / MIN_N_CLASS or use a bigger prediction file.")
    out_df = out_df.sort_values(["drug"]).reset_index(drop=True)

    os.makedirs("results/metrics", exist_ok=True)
    out_path = "results/metrics/per_drug_metrics_with_ci.csv"
    out_df.to_csv(out_path, index=False)

    print(f"✓ Wrote: {out_path}")
    print(f"✓ Drugs included: {len(out_df)}")
    print("Note: drugs with n<50 were skipped.")

if __name__ == "__main__":
    main()
