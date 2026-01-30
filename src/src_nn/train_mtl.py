# src_nn/train_mtl.py
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


class AMRDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        mask = (y != -1.0)
        return x, y, mask


def masked_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,T)
    y:     (B,T) values in {0,1} or -1 for missing
    mask:  (B,T) bool, True where label exists
    """
    # Replace missing labels with 0 to avoid NaNs (they'll be masked out)
    y_clean = torch.where(mask, y, torch.zeros_like(y))
    loss_raw = nn.functional.binary_cross_entropy_with_logits(
        logits, y_clean, reduction="none")  # (B,T)
    loss_masked = loss_raw[mask]
    if loss_masked.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return loss_masked.mean()


@torch.no_grad()
def eval_auroc(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_logits = []
    all_y = []
    for x, y, _mask in loader:
        x = x.to(device)
        logits = model(x).cpu()
        all_logits.append(logits)
        all_y.append(y.cpu())
    logits = torch.cat(all_logits, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()

    n_tasks = y.shape[1]
    per_task = []
    for t in range(n_tasks):
        yt = y[:, t]
        m = yt != -1
        if m.sum() < 10:
            per_task.append(np.nan)
            continue
        # need both classes present
        if len(np.unique(yt[m])) < 2:
            per_task.append(np.nan)
            continue
        per_task.append(roc_auc_score(yt[m], logits[m, t]))
    out = {
        "auroc_mean": float(np.nanmean(per_task)),
        "auroc_per_task": per_task,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", type=str, default=None,
                    help="Path to X .npy (features)")
    ap.add_argument("--y_path", type=str, default="data/ml/Y.npy",
                    help="Path to Y .npy (labels, missing=-1)")
    ap.add_argument("--antibiotics_path", type=str,
                    default="data/ml/antibiotics.txt")
    ap.add_argument("--out_dir", type=str, default="results")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--n_blocks", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--denoise", action="store_true",
                    help="Train with structured block dropout (denoising)")
    ap.add_argument("--block_size", type=int, default=256,
                    help="Feature block size for structured drop")
    ap.add_argument("--train_drop_frac", type=float, default=0.3,
                    help="Drop fraction during training if --denoise")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    x_path = args.x_path or autodetect_x_path()
    print(f"[INFO] Using X from: {x_path}")
    print(f"[INFO] Using Y from: {args.y_path}")

    X = np.load(x_path)
    Y = np.load(args.y_path, allow_pickle=True)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D (n_samples, n_features). Got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(
            f"Y must be 2D (n_samples, n_tasks). Got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y sample mismatch: {X.shape[0]} vs {Y.shape[0]}")

    n, d = X.shape
    t = Y.shape[1]
    print(f"[INFO] X shape: {X.shape}  Y shape: {Y.shape}")

    # simple split (you can replace with study-based later)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    train_ds = AMRDataset(X[tr], Y[tr])
    val_ds = AMRDataset(X[va], Y[va])
    test_ds = AMRDataset(X[te], Y[te])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    torch.manual_seed(args.seed)
    model = MultiTaskAMRNet(
        in_dim=d, n_tasks=t,
        hidden=args.hidden, emb_dim=args.emb_dim,
        n_blocks=args.n_blocks, dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    blocks = make_block_indices(d, args.block_size).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    best_val = -1.0
    best_state = None
    patience = 10
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            if args.denoise:
                x_in = block_dropout(
                    x, blocks, args.train_drop_frac, generator=gen)
            else:
                x_in = x

            logits = model(x_in)
            loss = masked_bce_with_logits(logits, y, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        # validation AUROC
        val_metrics = eval_auroc(model, val_loader, device=device)
        val_auc = val_metrics["auroc_mean"]

        print(
            f"[E{epoch:03d}] train_loss={total_loss/max(1, steps):.4f}  val_auroc={val_auc:.4f}")

        if val_auc > best_val + 1e-4:
            best_val = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("[INFO] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    val_out = eval_auroc(model, val_loader, device=device)
    test_out = eval_auroc(model, test_loader, device=device)

    # save metrics
    run_info = {
        "x_path": x_path,
        "y_path": args.y_path,
        "seed": args.seed,
        "denoise": bool(args.denoise),
        "block_size": args.block_size,
        "train_drop_frac": args.train_drop_frac if args.denoise else 0.0,
        "model": {
            "hidden": args.hidden,
            "emb_dim": args.emb_dim,
            "n_blocks": args.n_blocks,
            "dropout": args.dropout,
        },
        "val": val_out,
        "test": test_out,
    }
    tag = "mtl_denoise" if args.denoise else "mtl_scratch"
    with open(os.path.join(args.out_dir, f"{tag}_run.json"), "w") as f:
        json.dump(run_info, f, indent=2)


    # per-antibiotic CSV
    if os.path.exists(args.antibiotics_path):
        with open(args.antibiotics_path) as f:
            ab = [line.strip() for line in f if line.strip()]
        if len(ab) != t:
            print("[WARN] antibiotics.txt length mismatch, using generic names")
            ab = [f"ab_{i}" for i in range(t)]
    else:
        ab = [f"ab_{i}" for i in range(t)]

    df = pd.DataFrame({
        "antibiotic": ab,
        "val_auroc": val_out["auroc_per_task"],
        "test_auroc": test_out["auroc_per_task"],
    })
    df.to_csv(os.path.join(args.out_dir, f"{tag}_per_antibiotic.csv"), index=False)

    print(f"[DONE] Saved: {args.out_dir}/{tag}_run.json and {tag}_per_antibiotic.csv")
    print(f"[DONE] Val mean AUROC: {val_out['auroc_mean']:.4f}  Test mean AUROC: {test_out['auroc_mean']:.4f}")


if __name__ == "__main__":
    main()
