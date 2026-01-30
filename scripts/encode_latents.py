#!/usr/bin/env python3
import os, json, time
import numpy as np
import torch
from torch import nn

IN_DIR   = "kmers_proj/rp8192"
AE_PATH  = "models/ae/ae.pt"
OUT_DIR  = "latents/ae256"
META_OUT = os.path.join(OUT_DIR, "latents_meta.json")

LATENT  = 256
BATCH   = 2048
SEED    = 1337
DEVICE  = "cpu"

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
    def forward(self, x):
        return self.enc(x)

def encode_split(enc, split):
    in_path = os.path.join(IN_DIR, f"Z_{split}.npy")
    Z = np.load(in_path, mmap_mode="r")
    n, d = Z.shape

    out_path = os.path.join(OUT_DIR, f"L_{split}.npy")
    Lmm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, LATENT))

    device = torch.device(DEVICE)
    enc.to(device)
    enc.eval()

    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            xb = torch.from_numpy(np.asarray(Z[i:j], dtype=np.float32)).to(device)
            zb = enc(xb).cpu().numpy().astype(np.float32, copy=False)
            Lmm[i:j] = zb
            if i % (BATCH * 20) == 0:
                print(f"{split}: {j}/{n}")

    del Lmm
    print(f"✓ {split} saved -> {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Ztr = np.load(os.path.join(IN_DIR, "Z_train.npy"), mmap_mode="r")
    in_dim = Ztr.shape[1]

    enc = Encoder(in_dim, LATENT)
    sd = torch.load(AE_PATH, map_location="cpu")

    # Keep exact keys: enc.0.weight, enc.0.bias, ...
    enc_sd = {k: v for k, v in sd.items() if k.startswith("enc.")}
    enc.load_state_dict(enc_sd, strict=True)

    t0 = time.time()
    for split in ["train","val","test"]:
        encode_split(enc, split)

    meta = {
        "in_dir": IN_DIR,
        "ae_path": AE_PATH,
        "latent_dim": LATENT,
        "batch": BATCH,
        "seed": SEED,
        "device": DEVICE,
        "seconds": round(time.time() - t0, 2),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved: {META_OUT}")

if __name__ == "__main__":
    main()
