#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix


# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------
# Threshold selection (train only)
# ----------------------------
def best_threshold(y_true, probs):
    best_t = 0.5
    best_j = -1e9
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)


# ----------------------------
# Main
# ----------------------------
def main():
    Z = np.load("data/ml/Z.npy").astype(np.float32)
    Y = np.load("data/ml/Y.npy")
    antibiotics = [l.strip()
                   for l in open("data/ml/antibiotics.txt") if l.strip()]

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedShuffleSplit(
        n_splits=5, test_size=0.2, random_state=42)

    results = []

    print("Device:", device, flush=True)

    for j, ab in enumerate(antibiotics):
        y = Y[:, j]
        mask = y != -1
        Zj = Z[mask]
        yj = y[mask].astype(int)

        if len(np.unique(yj)) < 2 or len(yj) < 30:
            continue

        print(f"Training {ab} (n={len(yj)})", flush=True)

        aucs, sens, spec = [], [], []

        for tr_idx, te_idx in splitter.split(Zj, yj):
            Z_train, Z_test = Zj[tr_idx], Zj[te_idx]
            y_train, y_test = yj[tr_idx], yj[te_idx]

            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            pos_weight = torch.tensor(
                [neg / max(pos, 1)], dtype=torch.float32, device=device)

            model = MLP(Z_train.shape[1]).to(device)
            opt = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            Xtr = torch.tensor(Z_train, dtype=torch.float32, device=device)
            ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
            Xte = torch.tensor(Z_test, dtype=torch.float32, device=device)

            model.train()
            for _ in range(200):
                opt.zero_grad()
                logits = model(Xtr)
                loss = loss_fn(logits, ytr)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                logits_te = model(Xte).cpu().numpy()
                probs_te = 1 / (1 + np.exp(-logits_te))

                logits_tr = model(Xtr).cpu().numpy()
                probs_tr = 1 / (1 + np.exp(-logits_tr))

            aucs.append(roc_auc_score(y_test, probs_te))

            t = best_threshold(y_train, probs_tr)
            preds = (probs_te >= t).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
            spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

        results.append({
            "antibiotic": ab,
            "n": int(len(yj)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "sensitivity": float(np.mean(sens)),
            "specificity": float(np.mean(spec)),
        })

    df = pd.DataFrame(results).sort_values("auc_mean", ascending=False)
    df.to_csv("data/ml/mlp_results_fixed.csv", index=False)

    print("\n=== MLP RESULTS (FIXED) ===")
    print(df.to_string(index=False))
    print("\nSaved: data/ml/mlp_results_fixed.csv")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
