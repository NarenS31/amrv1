#!/usr/bin/env python3
import os
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from joblib import dump

KMER_DIR = "kmers/k31"
OUT_DIR  = "kmers_proj/rp8192"
DIM_OUT  = 8192
SEED     = 1337
BATCH    = 64

def load_X(split):
    return np.load(os.path.join(KMER_DIR, f"X_{split}.npy"), mmap_mode="r")

def project_split(rp, split):
    X = load_X(split)
    n, d = X.shape
    out_path = os.path.join(OUT_DIR, f"Z_{split}.npy")

    Zmm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n, DIM_OUT)
    )

    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        xb = np.asarray(X[i:j], dtype=np.float32)
        zb = rp.transform(xb).astype(np.float32, copy=False)
        Zmm[i:j] = zb
        if i % (BATCH * 50) == 0:
            print(f"{split}: {j}/{n}")

    del Zmm
    print(f"✓ {split} saved -> {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    Xtr = load_X("train")
    _, d = Xtr.shape

    rp = SparseRandomProjection(
        n_components=DIM_OUT,
        random_state=SEED,
        dense_output=True
    )

    rp.fit(np.zeros((1, d), dtype=np.float32))
    dump(rp, os.path.join(OUT_DIR, "rp.joblib"))
    print("✓ saved rp.joblib")

    for split in ["train", "val", "test"]:
        project_split(rp, split)

if __name__ == "__main__":
    main()
