#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
            nn.ReLU(),  # keep outputs non-negative-ish like counts
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="data/ml/X_reduced.npy")
    ap.add_argument("--out_dir", default="data/ml")
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    X = np.load(args.x).astype(np.float32)

    # simple log transform helps heavy-tailed k-mer counts
    X = np.log1p(X)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    x_tensor = torch.from_numpy(X)
    ds = TensorDataset(x_tensor)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, drop_last=False)

    model = Autoencoder(
        input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            xhat, _ = model(xb)
            loss = loss_fn(xhat, xb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d}  mse={total/n:.6f}")

    # Encode all samples
    model.eval()
    with torch.no_grad():
        Xd = x_tensor.to(device)
        _, Z = model(Xd)
        Z = Z.cpu().numpy().astype(np.float32)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(f"{args.out_dir}/Z.npy", Z)
    torch.save(model.state_dict(), f"{args.out_dir}/autoencoder.pt")

    print("\n=== AUTOENCODER DONE ===")
    print("Z shape:", Z.shape)
    print("Saved: data/ml/Z.npy")
    print("Saved: data/ml/autoencoder.pt")


if __name__ == "__main__":
    import os
    main()
