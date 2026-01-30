#!/usr/bin/env python3
import os, json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LAT_DIR="latents/ae256"
TASK_DIR="ml_tasks"
SEL_PATH="results/diffusion_task_selection.json"
OUT_ROOT="models/diffusion_latent_ddpm"

SEED=1337
DEVICE="cpu"

T=100
EPOCHS=40
BATCH=256
LR=2e-4
HID=512

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_betas(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class Denoiser(nn.Module):
    def __init__(self, xdim, hid, T, n_classes=2, tdim=128, ydim=32):
        super().__init__()
        self.T = T
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
    tr_idx = np.load(os.path.join(TASK_DIR, f"{drug}_train_idx.npy"))
    tr_y   = np.load(os.path.join(TASK_DIR, f"{drug}_train_y.npy"))
    return tr_idx, tr_y

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    set_seed(SEED)

    with open(SEL_PATH) as f:
        keep = json.load(f)["keep"]
    drugs = [k["drug"] for k in keep]
    if not drugs:
        raise SystemExit("No diffusion-eligible drugs.")

    Ltr = np.load(os.path.join(LAT_DIR, "L_train.npy"), mmap_mode="r")
    xdim = Ltr.shape[1]

    betas = make_betas(T).to(DEVICE)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)

    for drug in drugs:
        out_dir = os.path.join(OUT_ROOT, drug)
        os.makedirs(out_dir, exist_ok=True)
        ckpt_path = os.path.join(out_dir, "ddpm.pt")
        meta_path = os.path.join(out_dir, "train_meta.json")

        # Resume-safe: if model exists, skip training
        if os.path.exists(ckpt_path):
            print(f"\n=== SKIP (exists): {drug} ===")
            continue

        print(f"\n=== TRAIN DIFFUSION: {drug} ===")
        tr_idx, tr_y = load_task(drug)

        X = torch.from_numpy(np.asarray(Ltr[tr_idx], dtype=np.float32))
        y = torch.from_numpy(tr_y.astype(np.int64))
        dl = DataLoader(TensorDataset(X, y), batch_size=BATCH, shuffle=True)

        model = Denoiser(xdim=xdim, hid=HID, T=T).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)

        model.train()
        t0 = time.time()
        for ep in range(1, EPOCHS+1):
            total=0.0
            n=0
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                bsz = xb.size(0)
                t = torch.randint(0, T, (bsz,), device=DEVICE)

                eps = torch.randn_like(xb)
                a = abar[t].unsqueeze(1)
                x_t = torch.sqrt(a) * xb + torch.sqrt(1.0 - a) * eps

                eps_hat = model(x_t, t, yb)
                loss = (eps_hat - eps).pow(2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(loss.item()) * bsz
                n += bsz

            if ep % 5 == 0 or ep == 1:
                print(f"ep {ep:03d}/{EPOCHS}  mse={total/n:.6f}")

        ckpt = {
            "state_dict": model.state_dict(),  # torch save OK
            "xdim": int(xdim),
            "T": int(T),
            "HID": int(HID),
            "LR": float(LR),
            "EPOCHS": int(EPOCHS),
            "SEED": int(SEED),
            "beta_start": 1e-4,
            "beta_end": 0.02,
        }
        torch.save(ckpt, ckpt_path)

        meta = {
            "drug": drug,
            "seconds": round(time.time()-t0, 2),
            "xdim": int(xdim),
            "T": int(T),
            "HID": int(HID),
            "LR": float(LR),
            "EPOCHS": int(EPOCHS),
            "SEED": int(SEED),
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "batch": int(BATCH),
            "device": DEVICE,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ saved -> {ckpt_path}")
        print(f"✓ meta  -> {meta_path}")

if __name__ == "__main__":
    main()
