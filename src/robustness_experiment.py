#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix


# --- Autoencoder architecture must match what you trained ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


def encode_with_ae(X_reduced, ae_path, latent_dim=256):
    # log transform must match training
    X = np.log1p(X_reduced).astype(np.float32)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
    state = torch.load(ae_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        _, Z = model(Xt)
        Z = Z.detach().cpu().numpy().astype(np.float32)
    return Z


def eval_logreg_cv(X, y, n_splits=5, test_size=0.2, seed=42):
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed)
    aucs, sens, spec = [], [], []

    for tr, te in splitter.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)

        probs = clf.predict_proba(Xte)[:, 1]
        preds = (probs >= 0.5).astype(int)

        aucs.append(roc_auc_score(yte, probs))

        tn, fp, fn, tp = confusion_matrix(yte, preds).ravel()
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(sens)), float(np.mean(spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X_reduced", default="data/ml/X_reduced.npy")
    ap.add_argument("--Y", default="data/ml/Y.npy")
    ap.add_argument("--antibiotics", default="data/ml/antibiotics.txt")
    ap.add_argument("--ae_path", default="data/ml/autoencoder.pt")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--out_csv", default="data/ml/robustness_results.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X0 = np.load(args.X_reduced).astype(np.float32)
    Y = np.load(args.Y)
    antibiotics = [l.strip() for l in open(args.antibiotics) if l.strip()]

    rng = np.random.default_rng(args.seed)

    missing_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
    rows = []

    for miss in missing_levels:
        # mask features (same mask applied to all samples for this run)
        X = X0.copy()
        if miss > 0:
            d = X.shape[1]
            k = int(d * miss)
            drop = rng.choice(d, size=k, replace=False)
            X[:, drop] = 0.0

        # raw-logreg per antibiotic
        for j, ab in enumerate(antibiotics):
            y = Y[:, j]
            m = y != -1
            yj = y[m].astype(int)
            if len(np.unique(yj)) < 2 or len(yj) < 30:
                continue

            Xj = X[m]
            auc, auc_std, se, sp = eval_logreg_cv(Xj, yj)
            rows.append({"missing": miss, "model": "logreg_raw", "antibiotic": ab,
                         "n": int(len(yj)), "auc": auc, "auc_std": auc_std,
                         "sensitivity": se, "specificity": sp})

        # AE-logreg: encode masked X then logreg
        Z = encode_with_ae(X, args.ae_path, latent_dim=args.latent_dim)
        for j, ab in enumerate(antibiotics):
            y = Y[:, j]
            m = y != -1
            yj = y[m].astype(int)
            if len(np.unique(yj)) < 2 or len(yj) < 30:
                continue

            Zj = Z[m]
            auc, auc_std, se, sp = eval_logreg_cv(Zj, yj)
            rows.append({"missing": miss, "model": "logreg_ae", "antibiotic": ab,
                         "n": int(len(yj)), "auc": auc, "auc_std": auc_std,
                         "sensitivity": se, "specificity": sp})

        print(f"Done missing={miss}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("\nSaved:", args.out_csv)
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
