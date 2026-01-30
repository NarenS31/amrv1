import os
import numpy as np
import pandas as pd

X_PATH = "data/genomes/X_train.npy"
Y3_PATH = "data/genomes/y_train.npy"        # 0=S,1=I,2=R
AB_PATH = "data/genomes/ab_index.npy"
LABELS_PATH = "data/genomes/label_classes.npy"

OUT_DIR = "data/bin"
MIN_S = 15
MIN_NS = 15   # non-susceptible (I or R)

# mode: map I into R (non-susceptible)
MODE = "I_as_R"   # do NOT change

os.makedirs(OUT_DIR, exist_ok=True)

X = np.load(X_PATH).astype(np.float32)
y3 = np.load(Y3_PATH).astype(np.int64)
ab = np.load(AB_PATH).astype(np.int64)
labels = np.load(LABELS_PATH, allow_pickle=True)

if MODE != "I_as_R":
    raise ValueError("Only MODE=I_as_R supported here")

# Binary label: S=0, (I or R)=1
y = (y3 != 0).astype(np.int64)

# Filter antibiotics with both classes present
rows = []
keep_mask = np.zeros(len(y), dtype=bool)

for a in np.unique(ab):
    idx = np.where(ab == a)[0]
    s = int((y[idx] == 0).sum())
    ns = int((y[idx] == 1).sum())
    n = len(idx)
    rows.append((int(a), str(labels[a]), n, s, ns))
    if s >= MIN_S and ns >= MIN_NS:
        keep_mask[idx] = True

stats = pd.DataFrame(
    rows, columns=["ab_idx_old", "antibiotic", "n", "S", "NS"])
stats = stats.sort_values("n", ascending=False)
stats.to_csv(f"{OUT_DIR}/per_antibiotic_counts_before.csv", index=False)

X2, y2, ab2 = X[keep_mask], y[keep_mask], ab[keep_mask]

# Remap ab indices to 0..K-1
old_abs = np.unique(ab2)
old_to_new = {old: i for i, old in enumerate(old_abs)}
ab2 = np.array([old_to_new[a] for a in ab2], dtype=np.int64)
labels2 = np.array([labels[a] for a in old_abs], dtype=object)

np.save(f"{OUT_DIR}/X.npy", X2)
np.save(f"{OUT_DIR}/y.npy", y2)
np.save(f"{OUT_DIR}/ab.npy", ab2)
np.save(f"{OUT_DIR}/labels.npy", labels2)

# After-filter stats
rows2 = []
for new_a, name in enumerate(labels2):
    idx = np.where(ab2 == new_a)[0]
    s = int((y2[idx] == 0).sum())
    ns = int((y2[idx] == 1).sum())
    rows2.append((new_a, str(name), len(idx), s, ns))
pd.DataFrame(rows2, columns=["ab_idx", "antibiotic", "n", "S", "NS"]).sort_values("n", ascending=False)\
  .to_csv(f"{OUT_DIR}/per_antibiotic_counts_after.csv", index=False)

print("Saved binary dataset to data/bin/")
print("X:", X2.shape, "y:", y2.shape, "ab:",
      ab2.shape, "antibiotics:", len(labels2))
print("Class counts:", {0: int((y2 == 0).sum()), 1: int((y2 == 1).sum())})
