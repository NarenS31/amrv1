import glob
import numpy as np

files = sorted(glob.glob("data/genomes/*.npy") + glob.glob("data/ml/*.npy"))

print("=== NPYS + SHAPES ===")
for p in files:
    try:
        a = np.load(p, allow_pickle=True)
        print(f"{p:40s}  shape={a.shape}  dtype={a.dtype}")
    except Exception as e:
        print(f"{p:40s}  ERROR: {e}")
