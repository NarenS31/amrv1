#!/usr/bin/env python3
import os, json, glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

LAT_DIR="latents/ae256"
TASK_DIR="ml_tasks"
SEL_PATH="results/diffusion_task_selection.json"
DIFF_ROOT="models/diffusion_latent_ddpm"
OUT_DIR="results/diffusion_latents_v1"

SEED=1337
DEVICE="cpu"
TARGET_MINORITY_FRAC=0.40

def make_betas(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class Denoiser(nn.Module):
    def __init__(self, xdim, hid, T, n_classes=2, tdim=128, ydim=32):
        super().__init__()
        self.t_embed = nn.Embedding(T, tdim)
        self.y_embed = nn.Embedding(n_classes, ydim)
        self.net = nn.Sequential(
            nn.Linear(xdim + tdim + ydim, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, xdim),
        )
    def forward(self, x_t, t, y):
        te = self.t_embed(t)
        ye = self.y_embed(y)
        return self.net(torch.cat([x_t, te, ye], dim=1))

def load_task(drug):
    def p(suf): return os.path.join(TASK_DIR, f"{drug}_{suf}.npy")
    return (
        np.load(p("train_idx")), np.load(p("train_y")),
        np.load(p("val_idx")),   np.load(p("val_y")),
        np.load(p("test_idx")),  np.load(p("test_y")),
    )

def sgd_logreg(alpha):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=3000,
            tol=1e-4,
            class_weight="balanced",
            random_state=SEED
        ))
    ])

def eval_metrics(y, p):
    return {
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    }

@torch.no_grad()
def sample_ddpm(model, xdim, T, n, y_class, seed, beta_start, beta_end):
    torch.manual_seed(seed)
    betas = make_betas(T, beta_start, beta_end).to(DEVICE)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    x = torch.randn(n, xdim, device=DEVICE)
    y = torch.full((n,), int(y_class), device=DEVICE, dtype=torch.long)

    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=DEVICE, dtype=torch.long)
        eps_hat = model(x, tt, y)
        a_t = alphas[t]
        abar_t = abar[t]
        x = (1.0 / torch.sqrt(a_t)) * (x - ((1 - a_t) / torch.sqrt(1 - abar_t)) * eps_hat)
        if t > 0:
            x = x + torch.sqrt(betas[t]) * torch.randn_like(x)

    return x.cpu().numpy().astype(np.float32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(SEL_PATH) as f:
        drugs = [k["drug"] for k in json.load(f)["keep"]]

    Ltr = np.load(os.path.join(LAT_DIR, "L_train.npy"), mmap_mode="r")
    Lva = np.load(os.path.join(LAT_DIR, "L_val.npy"), mmap_mode="r")
    Lte = np.load(os.path.join(LAT_DIR, "L_test.npy"), mmap_mode="r")

    base_params = {}
    for p in glob.glob("results/latents_v1/*_best_params.json"):
        drug = os.path.basename(p).replace("_best_params.json","")
        with open(p) as f:
            base_params[drug] = float(json.load(f)["best_alpha"])

    rows=[]
    for drug in drugs:
        ckpt_path = os.path.join(DIFF_ROOT, drug, "ddpm.pt")
        if not os.path.exists(ckpt_path):
            print(f"SKIP {drug}: missing diffusion model")
            continue

        tr_idx, tr_y, va_idx, va_y, te_idx, te_y = load_task(drug)

        Xtr = np.asarray(Ltr[tr_idx], dtype=np.float32)
        Xva = np.asarray(Lva[va_idx], dtype=np.float32)
        Xte = np.asarray(Lte[te_idx], dtype=np.float32)

        n_pos = int(tr_y.sum())
        n_neg = int(len(tr_y) - n_pos)
        minority = 1 if n_pos < n_neg else 0
        n_min = min(n_pos, n_neg)
        n_tot = len(tr_y)
        target_min = int(TARGET_MINORITY_FRAC * n_tot)
        add_n = max(0, target_min - n_min)

        if add_n == 0:
            print(f"{drug}: no augmentation needed")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = Denoiser(
            xdim=ckpt["xdim"], hid=ckpt["HID"], T=ckpt["T"]
        ).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        synth = sample_ddpm(
            model, ckpt["xdim"], ckpt["T"], add_n,
            minority, SEED, ckpt["beta_start"], ckpt["beta_end"]
        )

        Xtr_aug = np.concatenate([Xtr, synth])
        ytr_aug = np.concatenate([tr_y, np.full(add_n, minority)])

        clf = sgd_logreg(base_params[drug])
        clf.fit(Xtr_aug, ytr_aug)

        va_p = clf.predict_proba(Xva)[:,1]
        te_p = clf.predict_proba(Xte)[:,1]

        rows.append({
            "drug": drug,
            "added_synth": add_n,
            "val_auprc_aug": eval_metrics(va_y, va_p)["auprc"],
            "test_auprc_aug": eval_metrics(te_y, te_p)["auprc"],
        })
        print(f"{drug}: +{add_n} synth  test_auprc={rows[-1]['test_auprc_aug']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "diffusion_aug_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved -> {out_csv}")
    print(df.sort_values("test_auprc_aug", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
