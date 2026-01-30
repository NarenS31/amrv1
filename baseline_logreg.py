import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

DATA = Path("data/ml")

Z = np.load(DATA / "Z_masked_ae.npy")
y = np.load(DATA / "Y.npy").reshape(-1)

print("Z:", Z.shape, "y:", y.shape)

# --- auto-detect an index array that matches y length ---
candidates = []

def safe_load_npy(path: Path):
    # Try standard load first; skip object arrays that require pickle
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as e:
        if "allow_pickle=False" in str(e) or "Object arrays cannot be loaded" in str(e):
            return None
        raise

for p in sorted(DATA.glob("*.npy")):
    name = p.name
    if name in {"Z_masked_ae.npy", "Z.npy", "Y.npy"}:
        continue

    arr = safe_load_npy(p)
    if arr is None:
        # object array or something unsafe; ignore for index detection
        continue

    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer) and len(arr) == len(y):
        candidates.append((name, arr))

if not candidates:
    print("\nNo matching index vector found (len == Y).")
    print("Listing safe-loadable .npy files:")
    for p in sorted(DATA.glob("*.npy")):
        a = safe_load_npy(p)
        if a is None:
            print(f" - {p.name:28s} SKIPPED (object array / requires pickle)")
        else:
            print(f" - {p.name:28s} shape={a.shape} dtype={a.dtype}")
    raise SystemExit(
        "\nFix: identify which file provides indices for Y.npy, or generate it during dataset creation."
    )

# Prefer common index filename if present
preferred = ["X_index.npy", "indices.npy", "labeled_indices.npy"]
picked = None
for pref in preferred:
    for name, arr in candidates:
        if name == pref:
            picked = (name, arr)
            break
    if picked:
        break
if not picked:
    picked = candidates[0]

idx_name, idx = picked
print(f"\nUsing index vector: {idx_name} (len={len(idx)})")
print("idx stats: min=", int(idx.min()), "max=", int(idx.max()), "unique=", int(len(np.unique(idx))))

Z_sub = Z[idx]
assert Z_sub.shape[0] == y.shape[0], f"Z_sub {Z_sub.shape[0]} != y {y.shape[0]}"

# --- split ---
X_train, X_test, y_train, y_test = train_test_split(
    Z_sub, y, test_size=0.2, random_state=42, stratify=y
)

# --- baseline model ---
clf = LogisticRegression(
    max_iter=5000,
    multi_class="multinomial",
    n_jobs=1
)
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)

# AUROC
auroc = np.nan
try:
    if len(np.unique(y)) == 2:
        auroc = roc_auc_score(y_test, probs[:, 1])
    else:
        auroc = roc_auc_score(y_test, probs, multi_class="ovr")
except Exception as e:
    print("AUROC failed:", e)

out = pd.DataFrame([{
    "method": "logreg_on_Z_masked_for_Y",
    "index_vector": idx_name,
    "accuracy": float(acc),
    "auroc": float(auroc) if np.isfinite(auroc) else np.nan,
    "n_total": int(len(y)),
    "n_test": int(len(y_test)),
    "random_state": 42
}])

Path("results").mkdir(exist_ok=True)
out.to_csv("results/baseline_logreg.csv", index=False)

print("\nSaved: results/baseline_logreg.csv")
print(out.to_string(index=False))

print("\nClassification report (test):")
print(classification_report(y_test, preds))
