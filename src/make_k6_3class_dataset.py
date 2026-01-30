#!/usr/bin/env python3
import numpy as np
import pandas as pd

X_PATH = "data/bvbrc/kmer_k6/X_k6.npy"
IDS_PATH = "data/bvbrc/kmer_k6/genome_ids_k6.txt"
PH_PATH = "data/bvbrc/amr_with_gcf_norm.csv"

OUT_X = "data/genomes/X_k6.npy"
OUT_Y = "data/genomes/y_k6_3class.npy"     # 0=S, 1=I, 2=R
OUT_AB = "data/genomes/ab_k6.npy"
OUT_LABS = "data/genomes/ab_labels_k6.npy"

# Load X + IDs
X = np.load(X_PATH)
with open(IDS_PATH) as f:
    ids = [line.strip() for line in f if line.strip()]

# IDs are filenames like GCF_XXXX..._genomic ; make accession key = first token up to version


def to_gcf_key(s: str) -> str:
    # expected something starting with GCF_#########.#
    # keep first two chunks like GCF_001234567.1
    parts = s.split("_")
    # if filename is exactly accession, return it
    if s.startswith("GCF_") and "." in s and len(parts) >= 2 and parts[1].count(".") == 1:
        # e.g. GCF_001741545.1_ASM... -> "GCF_001741545.1"
        return parts[0] + "_" + parts[1]
    if s.startswith("GCF_") and "." in s:
        return s.split("_genomic")[0]
    return s


gcf_keys = [to_gcf_key(s) for s in ids]
key_to_row = {k: i for i, k in enumerate(gcf_keys)}

# Load phenotypes (already has gcf, Antibiotic, p)
ph = pd.read_csv(PH_PATH, low_memory=False)
ph["gcf"] = ph["gcf"].astype(str).str.strip()
ph["Antibiotic"] = ph["Antibiotic"].astype(str).str.strip().str.lower()

pmap = {"S": 0, "I": 1, "R": 2}
ph = ph[ph["p"].isin(pmap)]
ph["y"] = ph["p"].map(pmap).astype(np.int8)

# keep only rows whose gcf we have in X
ph = ph[ph["gcf"].isin(key_to_row)].copy()
print("[INFO] phenotype rows matched to kmer X:", len(ph))

# Build sample rows: each (genome, antibiotic) becomes one sample
rows = []
for _, r in ph.iterrows():
    rows.append((key_to_row[r["gcf"]], r["Antibiotic"], r["y"]))

# Antibiotic index mapping
ab_labels = sorted({ab for _, ab, _ in rows})
ab2i = {ab: i for i, ab in enumerate(ab_labels)}

# Final arrays
N = len(rows)
X_out = np.zeros((N, X.shape[1]), dtype=np.uint32)
y_out = np.zeros((N,), dtype=np.int8)
ab_out = np.zeros((N,), dtype=np.int16)

for i, (row, ab, y) in enumerate(rows):
    X_out[i] = X[row]
    y_out[i] = y
    ab_out[i] = ab2i[ab]

np.save(OUT_X, X_out)
np.save(OUT_Y, y_out)
np.save(OUT_AB, ab_out)
np.save(OUT_LABS, np.array(ab_labels, dtype=object))

print("[DONE] X:", X_out.shape, X_out.dtype)
print("[DONE] y:", y_out.shape, "classes:", dict(
    zip(*np.unique(y_out, return_counts=True))))
print("[DONE] ab:", ab_out.shape, "unique antibiotics:", len(ab_labels))
print("[DONE] saved labels ->", OUT_LABS)
