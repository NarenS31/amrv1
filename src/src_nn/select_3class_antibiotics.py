# save as src/src_nn/select_3class_antibiotics.py
import numpy as np
from collections import Counter, defaultdict

ab = np.load("data/genomes/ab_index.npy")
y = np.load("data/genomes/y_train.npy")
labels = np.load("data/genomes/label_classes.npy", allow_pickle=True)

d = defaultdict(Counter)
for yi, abi in zip(y, ab):
    d[int(abi)][int(yi)] += 1

rows = []
for abi, cnt in d.items():
    s, i, r = cnt[0], cnt[1], cnt[2]
    rows.append((s, i, r, int(abi), str(labels[abi])))

rows.sort(key=lambda x: (x[1], x[0]+x[2]), reverse=True)

print("S  I  R   antibiotic")
for s, i, r, abi, name in rows[:50]:
    print(f"{s:3d}{i:3d}{r:3d}  {name}")

# pick thresholds
MIN_S, MIN_I, MIN_R = 30, 10, 30
keep = [abi for s, i, r, abi, name in rows if s >=
        MIN_S and i >= MIN_I and r >= MIN_R]

print("\nKept:", len(keep))
print([str(labels[k]) for k in keep[:30]])

np.save("data/genomes/ab_keep_3class.npy", np.array(keep, dtype=np.int32))
print("Saved data/genomes/ab_keep_3class.npy")
