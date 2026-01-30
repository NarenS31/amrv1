from __future__ import annotations
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score


from .corruption import make_block_indices, block_dropout
from .model_conditional import ConditionalAMRNet


class CondDataset(Dataset):
    def __init__(self, X, y, ab):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.ab = ab.astype(np.int64, copy=False)

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.y[i], dtype=torch.long),
            torch.tensor(self.ab[i], dtype=torch.long),
        )


@torch.no_grad()
def eval_metrics(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y, ab in loader:
        x, y, ab = x.to(device), y.to(device), ab.to(device)
        logits = model(x, ab)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        "bal_acc": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", default="data/genomes/X_train.npy")
    ap.add_argument("--y_path", default="data/genomes/y_train.npy")
    ap.add_argument("--ab_path", default="data/genomes/ab_index.npy")
    ap.add_argument("--labels_path", default="data/genomes/label_classes.npy")
    ap.add_argument("--out_dir", default="results")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--ab_emb_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--train_drop_frac", type=float, default=0.3)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X = np.load(args.x_path)
    y = np.load(args.y_path)
    ab = np.load(args.ab_path)
    labels = np.load(args.labels_path, allow_pickle=True)
    n_antibiotics = len(labels)

    if X.shape[0] != y.shape[0] or X.shape[0] != ab.shape[0]:
        raise ValueError(f"Mismatch: X={X.shape}, y={y.shape}, ab={ab.shape}")

    n, d = X.shape
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    train_ds = CondDataset(X[tr], y[tr], ab[tr])
    val_ds = CondDataset(X[va], y[va], ab[va])
    test_ds = CondDataset(X[te], y[te], ab[te])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device, "X:",
          X.shape, "antibiotics:", n_antibiotics)

    torch.manual_seed(args.seed)
    model = ConditionalAMRNet(
        in_dim=d,
        n_antibiotics=n_antibiotics,
        ab_emb_dim=args.ab_emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    blocks = make_block_indices(d, args.block_size).to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    best = -1.0
    best_state = None
    bad = 0
    patience = 10

    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.0
        steps = 0

        for x, yb, abb in train_loader:
            x, yb, abb = x.to(device), yb.to(device), abb.to(device)

            if args.denoise:
                x = block_dropout(
                    x, blocks, args.train_drop_frac, generator=gen)

            logits = model(x, abb)
            loss = ce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            steps += 1

        val = eval_metrics(model, val_loader, device)
        score = val["macro_f1"]
        print(f"[E{epoch:03d}] loss={total/max(1, steps):.4f}  val_bal_acc={val['bal_acc']:.4f}  val_macro_f1={val['macro_f1']:.4f}")

        if score > best + 1e-4:
            best = score
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

    val = eval_metrics(model, val_loader, device)
    test = eval_metrics(model, test_loader, device)

    tag = "cond_denoise" if args.denoise else "cond"
    out = {
        "x_path": args.x_path,
        "y_path": args.y_path,
        "ab_path": args.ab_path,
        "labels_path": args.labels_path,
        "seed": args.seed,
        "denoise": bool(args.denoise),
        "block_size": args.block_size,
        "train_drop_frac": args.train_drop_frac if args.denoise else 0.0,
        "model": {"hidden": args.hidden, "ab_emb_dim": args.ab_emb_dim, "dropout": args.dropout},
        "val": val,
        "test": test,
    }
    with open(os.path.join(args.out_dir, f"{tag}_run.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] saved results/{tag}_run.json")
    print(f"[DONE] val={val}  test={test}")


if __name__ == "__main__":
    main()
