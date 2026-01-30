#!/usr/bin/env python3
import os, json, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

LAT_DIR   = "latents/ae256"
TASK_DIR  = "ml_tasks"
OUT_DIR   = "results/latents_v1"

ALPHAS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
MAX_ITERS = 3000
SEED = 1337

def load_lat(split):
    return np.load(os.path.join(LAT_DIR, f"L_{split}.npy"), mmap_mode="r")

def load_task(drug):
    def p(suf): return os.path.join(TASK_DIR, f"{drug}_{suf}.npy")
    tr_idx = np.load(p("train_idx"))
    tr_y   = np.load(p("train_y"))
    va_idx = np.load(p("val_idx"))
    va_y   = np.load(p("val_y"))
    te_idx = np.load(p("test_idx"))
    te_y   = np.load(p("test_y"))
    return tr_idx, tr_y, va_idx, va_y, te_idx, te_y

def metrics(y_true, y_score, y_pred):
    out = {}
    out["auroc"] = float("nan")
    try:
        if len(np.unique(y_true)) > 1:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    out["auprc"] = float(average_precision_score(y_true, y_score))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred))
    out["prevalence"] = float(np.mean(y_true))
    out["n"] = int(len(y_true))
    out["n_pos"] = int(np.sum(y_true))
    out["n_neg"] = int(len(y_true) - np.sum(y_true))
    return out

def make_model(alpha):
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=MAX_ITERS,
            tol=1e-4,
            class_weight="balanced",
            random_state=SEED
        ))
    ])

def is_degenerate(y):
    u = np.unique(y)
    return len(u) < 2

def train_eval_one(drug, Ltr, Lva, Lte):
    tr_idx, tr_y, va_idx, va_y, te_idx, te_y = load_task(drug)

    # Degenerate splits = skip (can’t train or evaluate honestly)
    deg = []
    if is_degenerate(tr_y): deg.append("train")
    if is_degenerate(va_y): deg.append("val")
    if is_degenerate(te_y): deg.append("test")
    if deg:
        return None, {"reason": "single_class_split", "degenerate": deg,
                      "train_classes": np.unique(tr_y).tolist(),
                      "val_classes": np.unique(va_y).tolist(),
                      "test_classes": np.unique(te_y).tolist()}

    Xtr = np.asarray(Ltr[tr_idx], dtype=np.float32)
    Xva = np.asarray(Lva[va_idx], dtype=np.float32)
    Xte = np.asarray(Lte[te_idx], dtype=np.float32)

    best_alpha = None
    best_val_auprc = -1.0

    for a in ALPHAS:
        m = make_model(a)
        m.fit(Xtr, tr_y)
        va_proba = m.predict_proba(Xva)[:, 1]
        va_pred  = (va_proba >= 0.5).astype(int)
        va_m = metrics(va_y, va_proba, va_pred)
        if va_m["auprc"] > best_val_auprc:
            best_val_auprc = va_m["auprc"]
            best_alpha = a

    model = make_model(best_alpha)
    model.fit(Xtr, tr_y)

    va_proba = model.predict_proba(Xva)[:, 1]
    va_pred  = (va_proba >= 0.5).astype(int)
    te_proba = model.predict_proba(Xte)[:, 1]
    te_pred  = (te_proba >= 0.5).astype(int)

    val_m = metrics(va_y, va_proba, va_pred)
    tst_m = metrics(te_y, te_proba, te_pred)

    out = {
        "drug": drug,
        "best_alpha": best_alpha,
        **{f"val_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in tst_m.items()},
    }
    meta = {"best_alpha": best_alpha, "alphas": ALPHAS}
    return out, meta

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    Ltr = load_lat("train")
    Lva = load_lat("val")
    Lte = load_lat("test")

    drugs = sorted(set(
        os.path.basename(p).replace("_train_idx.npy","")
        for p in glob.glob(os.path.join(TASK_DIR, "*_train_idx.npy"))
    ))
    if not drugs:
        raise SystemExit(f"No tasks found in {TASK_DIR} (expected *_train_idx.npy)")

    rows = []
    skipped = {}

    for d in drugs:
        print(f"== {d} ==")
        r, meta = train_eval_one(d, Ltr, Lva, Lte)
        if r is None:
            skipped[d] = meta
            print(f"  SKIP: {meta}")
            continue

        rows.append(r)
        with open(os.path.join(OUT_DIR, f"{d}_best_params.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  val_auprc={r['val_auprc']:.4f}  test_auprc={r['test_auprc']:.4f}  best_alpha={r['best_alpha']}")

    if rows:
        df = pd.DataFrame(rows).sort_values("test_auprc", ascending=False)
        out_csv = os.path.join(OUT_DIR, "latent_logreg_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n✓ Saved: {out_csv}")
        print(df[["drug","val_auprc","test_auprc","val_auroc","test_auroc","test_prevalence","test_n","test_n_pos","test_n_neg","best_alpha"]].to_string(index=False))
    else:
        print("No non-degenerate tasks produced results.")

    skip_path = os.path.join(OUT_DIR, "skipped_tasks.json")
    with open(skip_path, "w") as f:
        json.dump(skipped, f, indent=2)
    print(f"\n✓ Saved: {skip_path} (skipped degenerate tasks)")

if __name__ == "__main__":
    main()
