
import numpy as np
from collections import Counter, defaultdict

Y = "data/genomes/y_k6_3class.npy"
AB = "data/genomes/ab_k6.npy"
LABS = "data/genomes/ab_labels_k6.npy"

# you can tune these
MIN_I = 30
MIN_TOTAL = 300

y = np.load(Y)
ab = np.load(AB)
labs = np.load(LABS, allow_pickle=True)

d = defaultdict(Counter)
for yi, abi in zip(y, ab):
    d[int(abi)][int(yi)] += 1

keep3 = []
rows = []
for abi, c in d.items():
    s, i, r = c[0], c[1], c[2]
    n = s+i+r
    name = str(labs[abi])
    rows.append((n, s, i, r, name))
    if i >= MIN_I and n >= MIN_TOTAL and s > 0 and r > 0:
        keep3.append(int(abi))

rows.sort(reverse=True)

print("Top 30 antibiotics:")
for n, s, i, r, name in rows[:30]:
    print(f"{name:35s} n={n:5d}  S={s:5d} I={i:5d} R={r:5d}")

print("\n3-class eligible:", len(keep3))
print("eligible names:", [str(labs[i]) for i in keep3][:100])

out = "data/genomes/ab_keep_3class_hybrid.npy"
np.save(out, np.array(keep3, dtype=np.int16))
print("[OK] saved", out)
