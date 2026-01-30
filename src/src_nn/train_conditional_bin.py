from __future__ import annotations
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# -------------------------
# Dataset
# -------------------------


class CondBinDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ab: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)  # BCE wants float
        self.ab = ab.astype(np.int64, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.float32),
            torch.tensor(self.ab[i], dtype=torch.long),
        )

# -------------------------
# Model (conditional)
# -------------------------


class ConditionalAMRNetBin(nn.Module):
    def __init__(self, in_dim: int, n_antibiotics: int, ab_emb_dim: int = 32,
                 hidden: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.ab_emb = nn.Embedding(n_antibiotics, ab_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim + ab_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),  # single logit
        )

    def forward(self, x, ab_idx):
        e = self.ab_emb(ab_idx)
        h = torch.cat([x, e], dim=1)
        return self.net(h).squeeze(1)  # (B,)


@torch.no_grad()
def eval_auc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    ys, ps = [], []
    for x, y, ab in loader:
        x = x.to(device)
        ab = ab.to(device)
        logits = model(x, ab).detach().cpu()
        p = torch.sigmoid(logits)
        ys.append(y.cpu())
        ps.append(p)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    # need both classes present
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", default="data/bin/X.npy")
    ap.add_argument("--y_path", default="data/bin/y.npy")
    ap.add_argument("--ab_path", default="data/bin/ab.npy")
    ap.add_argument("--labels_path", default="data/bin/labels.npy")
    ap.add_argument("--out", default="results/cond_bin_run.json")

    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--ab_emb_dim", type=int, default=32)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X = np.load(args.x_path)
    y = np.load(args.y_path)
    ab = np.load(args.ab_path)
    labels = np.load(args.labels_path, allow_pickle=True)
    n, d = X.shape
    k = int(ab.max()) + 1

    assert n == len(y) == len(ab)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}  X: {X.shape}  antibiotics: {k}")

    # split
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    tr = idx[:n_train]
    va = idx[n_train:n_train+n_val]
    te = idx[n_train+n_val:]

    train = CondBinDataset(X[tr], y[tr], ab[tr])
    val = CondBinDataset(X[va], y[va], ab[va])
    test = CondBinDataset(X[te], y[te], ab[te])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)

    model = ConditionalAMRNetBin(
        in_dim=d, n_antibiotics=k,
        ab_emb_dim=args.ab_emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    # BCE pos_weight for imbalance
    ytr = y[tr]
    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    best = -1.0
    best_state = None
    patience = 12
    bad = 0

    for e in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0

        for xb, yb, abb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            abb = abb.to(device)

            logits = model(xb, abb)
            loss = crit(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            steps += 1

        val_auc = eval_auc(model, val_loader, device)
        print(f"[E{e:03d}] loss={total/max(steps, 1):.4f}  val_auc={val_auc:.4f}")

        if val_auc > best + 1e-4:
            best = val_auc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("[INFO] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_auc = eval_auc(model, val_loader, device)
    test_auc = eval_auc(model, test_loader, device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {
        "x": args.x_path,
        "y": args.y_path,
        "ab": args.ab_path,
        "n": int(n),
        "d": int(d),
        "k_antibiotics": int(k),
        "pos_weight": float(pos_weight.item()),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "labels_preview": [str(x) for x in labels[:10]],
        "model": {"hidden": args.hidden, "dropout": args.dropout, "ab_emb_dim": args.ab_emb_dim},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] saved {args.out}")
    print(f"[DONE] val_auc={val_auc:.4f}  test_auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
