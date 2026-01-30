#!/usr/bin/env python3
import os, json, time
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# XGBoost optional
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

LAT_DIR = "latents/ae256"
TASK_DIR = "ml_tasks"
OUT_DIR = "results/stronger_v1"
SEED = 1337

def load_latents():
    Ltr = np.load(os.path.join(LAT_DIR, "L_train.npy"), mmap_mode="r")
    Lva = np.load(os.path.join(LAT_DIR, "L_val.npy"), mmap_mode="r")
    Lte = np.load(os.path.join(LAT_DIR, "L_test.npy"), mmap_mode="r")
    return Ltr, Lva, Lte

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

def load_task(drug):
    def p(name): return os.path.join(TASK_DIR, f"{drug}_{name}.npy")
    tr_idx = np.load(p("train_idx")).astype(int)
    va_idx = np.load(p("val_idx")).astype(int)
    te_idx = np.load(p("test_idx")).astype(int)
    tr_y = np.load(p("train_y")).astype(int)
    va_y = np.load(p("val_y")).astype(int)
    te_y = np.load(p("test_y")).astype(int)
    return (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y)

def get_drug_list():
    # task jsons exist: ml_tasks/<drug>.json
    drugs = []
    for fn in os.listdir(TASK_DIR):
        if fn.endswith(".json") and not fn.endswith("_meta.json"):
            drugs.append(fn.replace(".json",""))
    drugs = sorted(set(drugs))
    return drugs

def train_eval_svm(trX, trY, vaX, vaY, teX, teY):
    """
    LinearSVC + Platt scaling (logistic calibration) using VAL only.
    This avoids sklearn's cv='prefit' incompatibilities.
    """
    Cs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]
    best = None

    for C in Cs:
        # Fit scaler + LinearSVC on TRAIN
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xtr_s = scaler.fit_transform(trX)
        Xva_s = scaler.transform(vaX)
        Xte_s = scaler.transform(teX)

        svc = LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=20000)
        svc.fit(Xtr_s, trY)

        # Platt scaling: fit logistic regression on VAL using decision_function outputs
        va_score = svc.decision_function(Xva_s).reshape(-1, 1)
        te_score = svc.decision_function(Xte_s).reshape(-1, 1)

        # If val is degenerate, skip
        if len(np.unique(vaY)) < 2:
            continue

        cal = LogisticRegression(solver="lbfgs", max_iter=2000)
        cal.fit(va_score, vaY)

        va_prob = cal.predict_proba(va_score)[:, 1]
        va_auprc = average_precision_score(vaY, va_prob)

        if (best is None) or (va_auprc > best["val_auprc"]):
            te_prob = cal.predict_proba(te_score)[:, 1]
            best = {
                "C": C,
                "val_auprc": float(va_auprc),
                "val_auroc": safe_auc(vaY, va_prob),
                "test_auprc": float(average_precision_score(teY, te_prob)),
                "test_auroc": safe_auc(teY, te_prob),
                "val_prob": va_prob.astype(np.float32),
                "test_prob": te_prob.astype(np.float32),
            }

    if best is None:
        return {"C": None, "val_auprc": float("nan"), "val_auroc": float("nan"),
                "test_auprc": float("nan"), "test_auroc": float("nan"),
                "val_prob": np.array([], dtype=np.float32),
                "test_prob": np.array([], dtype=np.float32)}
    return best

def train_eval_xgb(trX, trY, vaX, vaY, teX, teY):
    if not HAVE_XGB:
        return {"error": "xgboost not installed"}

    # Very small grid; do NOT waste time. We tune on val AUPRC.
    # Handle imbalance with scale_pos_weight computed from train only.
    pos = trY.sum()
    neg = len(trY) - pos
    spw = float(neg / max(pos, 1))

    grid = [
        dict(max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(max_depth=3, eta=0.10, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
        dict(max_depth=5, eta=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0),
    ]

    dtr = xgb.DMatrix(trX, label=trY)
    dva = xgb.DMatrix(vaX, label=vaY)
    dte = xgb.DMatrix(teX, label=teY)

    best = None
    for params in grid:
        p = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": SEED,
            "verbosity": 0,
            "nthread": 4,
            "scale_pos_weight": spw,
            **params,
        }
        # early stopping on val, train-only data
        bst = xgb.train(
            p, dtr,
            num_boost_round=2000,
            evals=[(dva, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        va_prob = bst.predict(dva)
        va_auprc = average_precision_score(vaY, va_prob)

        if (best is None) or (va_auprc > best["val_auprc"]):
            te_prob = bst.predict(dte)
            best = {
                "params": p,
                "best_iteration": int(bst.best_iteration),
                "val_auprc": float(va_auprc),
                "val_auroc": safe_auc(vaY, va_prob),
                "test_auprc": float(average_precision_score(teY, te_prob)),
                "test_auroc": safe_auc(teY, te_prob),
                "val_prob": va_prob.astype(np.float32),
                "test_prob": te_prob.astype(np.float32),
                "model": bst,
            }
    return best

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Ltr, Lva, Lte = load_latents()
    drugs = get_drug_list()

    out_rows = []
    skipped = {}

    t0_all = time.time()
    for drug in drugs:
        (tr_idx, tr_y), (va_idx, va_y), (te_idx, te_y) = load_task(drug)

        # guard degenerate tasks
        if len(np.unique(tr_y)) < 2 or len(np.unique(va_y)) < 2 or len(np.unique(te_y)) < 2:
            skipped[drug] = {"reason": "degenerate split (single class in one split)"}
            continue

        trX = np.asarray(Ltr[tr_idx], dtype=np.float32)
        vaX = np.asarray(Lva[va_idx], dtype=np.float32)
        teX = np.asarray(Lte[te_idx], dtype=np.float32)

        print(f"\n== {drug} ==")

        # ---- SVM ----
        svm_best = train_eval_svm(trX, tr_y, vaX, va_y, teX, te_y)
        out_rows.append({
            "drug": drug,
            "model": "svm_calibrated",
            "val_auprc": svm_best["val_auprc"],
            "test_auprc": svm_best["test_auprc"],
            "val_auroc": svm_best["val_auroc"],
            "test_auroc": svm_best["test_auroc"],
            "best_C": svm_best["C"],
            "test_n": int(len(te_y)),
            "test_prev": float(te_y.mean()),
            "test_pos": int(te_y.sum()),
            "test_neg": int(len(te_y) - te_y.sum()),
        })
        np.save(os.path.join(OUT_DIR, f"{drug}_svm_val_prob.npy"), svm_best["val_prob"])
        np.save(os.path.join(OUT_DIR, f"{drug}_svm_test_prob.npy"), svm_best["test_prob"])
        print(f"  SVM:  val_auprc={svm_best['val_auprc']:.4f}  test_auprc={svm_best['test_auprc']:.4f}  C={svm_best['C']}")

        # ---- XGB ----
        xgb_best = train_eval_xgb(trX, tr_y, vaX, va_y, teX, te_y)
        if "error" in xgb_best:
            print(f"  XGB:  SKIP ({xgb_best['error']})")
        else:
            out_rows.append({
                "drug": drug,
                "model": "xgboost",
                "val_auprc": xgb_best["val_auprc"],
                "test_auprc": xgb_best["test_auprc"],
                "val_auroc": xgb_best["val_auroc"],
                "test_auroc": xgb_best["test_auroc"],
                "best_iteration": xgb_best["best_iteration"],
                "max_depth": xgb_best["params"]["max_depth"],
                "eta": xgb_best["params"]["eta"],
                "subsample": xgb_best["params"]["subsample"],
                "colsample_bytree": xgb_best["params"]["colsample_bytree"],
                "scale_pos_weight": xgb_best["params"]["scale_pos_weight"],
                "test_n": int(len(te_y)),
                "test_prev": float(te_y.mean()),
                "test_pos": int(te_y.sum()),
                "test_neg": int(len(te_y) - te_y.sum()),
            })
            np.save(os.path.join(OUT_DIR, f"{drug}_xgb_val_prob.npy"), xgb_best["val_prob"])
            np.save(os.path.join(OUT_DIR, f"{drug}_xgb_test_prob.npy"), xgb_best["test_prob"])
            # save model
            xgb_best["model"].save_model(os.path.join(OUT_DIR, f"{drug}_xgb.json"))
            print(f"  XGB:  val_auprc={xgb_best['val_auprc']:.4f}  test_auprc={xgb_best['test_auprc']:.4f}  iter={xgb_best['best_iteration']}")

    df = pd.DataFrame(out_rows)
    out_csv = os.path.join(OUT_DIR, "stronger_latent_results.csv")
    df.to_csv(out_csv, index=False)

    with open(os.path.join(OUT_DIR, "skipped_tasks.json"), "w") as f:
        json.dump(skipped, f, indent=2)

    print("\n✓ Saved:", out_csv)
    print("✓ Saved:", os.path.join(OUT_DIR, "skipped_tasks.json"))
    print("seconds:", round(time.time()-t0_all, 2))

if __name__ == "__main__":
    main()
