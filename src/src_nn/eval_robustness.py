# src_nn/eval_robustness.py
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score

from .corruption import make_block_indices, block_dropout
from .model_mtl import MultiTaskAMRNet


def autodetect_x_path() -> str:
    candidates = [
        "data/genomes/kmer_matrix.npy",
        "data/genomes/kmer_features.npy",
        "data/genomes/X_train.npy",
        "data/ml/X_index.npy",
        "data/ml/X_reduced.npy",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not autodetect X .npy file. Pass --x_path explicitly.")


@torch.no_grad()
def predict_logits(model, X, device, batch_size=64):
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", type=str, default=None)
    ap.add_argument("--y_path", type=str, default="data/ml/Y.npy")
    ap.add_argument("--antibiotics_path", type=str,
                    default="data/ml/antibiotics.txt")
    ap.add_argument("--out_csv", type=str,
                    default="results/robustness_curves_nn.csv")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--fractions", type=str, default="0,0.1,0.2,0.4,0.6")
    ap.add_argument("--weights", type=str, default="",
                    help="Optional path to trained model .pt")

    args = ap.parse_args()

    x_path = args.x_path or autodetect_x_path()
    X = np.load(x_path).astype(np.float32, copy=False)
    Y = np.load(args.y_path, allow_pickle=True).astype(np.float32, copy=False)

    n, d = X.shape
    t = Y.shape[1]

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskAMRNet(
        in_dim=d, n_tasks=t,
        hidden=args.hidden, emb_dim=args.emb_dim,
        n_blocks=args.n_blocks, dropout=args.dropout
    ).to(device)

    if args.weights and os.path.exists(args.weights):
        sd = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(sd)
        print(f"[INFO] Loaded weights: {args.weights}")
    else:
        print(
            "[WARN] No weights provided; evaluating randomly initialized model (not useful).")

    blocks = make_block_indices(d, args.block_size).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    fracs = [float(x.strip()) for x in args.fractions.split(",")]


if os.path.exists(args.antibiotics_path):
    with open(args.antibiotics_path) as f:
        ab = [line.strip() for line in f if line.strip()]
    if len(ab) != t:
        print("[WARN] antibiotics.txt length mismatch, using generic names")
        ab = [f"ab_{i}" for i in range(t)]
else:
    ab = [f"ab_{i}" for i in range(t)]

    rows = []
    for f in fracs:
        # apply structured dropout at eval time
        Xc = []
        bs = 256  # cpu side batching for corruption
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device)
            xb2 = block_dropout(xb, blocks, f, generator=gen).cpu().numpy()
            Xc.append(xb2)
        Xc = np.concatenate(Xc, axis=0)

        logits = predict_logits(model, Xc, device=device, batch_size=64)

        # compute per-antibiotic AUROC ignoring missing labels
        per_task = []
        for j in range(t):
            yj = Y[:, j]
            m = yj != -1
            if m.sum() < 10 or len(np.unique(yj[m])) < 2:
                per_task.append(np.nan)
            else:
                per_task.append(roc_auc_score(yj[m], logits[m, j]))

        rows.append({
            "missing": f,
            "auroc_mean": float(np.nanmean(per_task)),
            **{ab[j]: per_task[j] for j in range(t)}
        })
        print(f"[missing={f:.1f}] mean AUROC={np.nanmean(per_task):.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[DONE] Saved {args.out_csv}")


if __name__ == "__main__":
    main()
