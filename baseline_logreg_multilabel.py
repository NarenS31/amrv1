import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

DATA = Path("data/ml")

Z = np.load(DATA / "Z.npy")              # (187, 256)
Y = np.load(DATA / "Y.npy")              # (187, 15) multi-label (0/1)
X_index = np.load(DATA / "X_index.npy")  # (187,) just for provenance

print("Z:", Z.shape, "Y:", Y.shape, "X_index:", X_index.shape)

assert Z.shape[0] == Y.shape[0], "Z and Y row mismatch"

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    Z, Y, test_size=0.2, random_state=42
)

# One-vs-rest logistic regression for multi-label
base = LogisticRegression(max_iter=5000)
clf = OneVsRestClassifier(base)
clf.fit(X_train, Y_train)

probs = clf.predict_proba(X_test)  # shape (n_test, 15)

# AUROC: micro + macro
micro = roc_auc_score(Y_test, probs, average="micro")
macro = roc_auc_score(Y_test, probs, average="macro")

out = pd.DataFrame([{
    "method": "OVR_logreg_on_Z_multi_label",
    "n_total": int(Z.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_labels": int(Y.shape[1]),
    "auroc_micro": float(micro),
    "auroc_macro": float(macro),
    "random_state": 42
}])

Path("results").mkdir(exist_ok=True)
out.to_csv("results/baseline_logreg_multilabel.csv", index=False)

print("\nSaved: results/baseline_logreg_multilabel.csv")
print(out.to_string(index=False))
