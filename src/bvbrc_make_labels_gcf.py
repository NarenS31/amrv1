#!/usr/bin/env python3
import pandas as pd

GENOMES = "data/bvbrc/kp_genomes.csv"
AMR = "data/bvbrc/kp_amr.csv"
MAP = "data/bvbrc/gca_to_gcf.csv"
OUT = "data/bvbrc/amr_with_gcf.csv"


def norm_pheno(x: str):
    if not isinstance(x, str):
        return None
    x = x.strip().lower()
    m = {
        "susceptible": "S",
        "resistant": "R",
        "intermediate": "I",
        "susceptible-dose dependent": "I",   # treat as I
        "nonsusceptible": "R",
    }
    return m.get(x, None)


g = pd.read_csv(GENOMES, low_memory=False)
a = pd.read_csv(AMR, low_memory=False)
m = pd.read_csv(MAP, low_memory=False)   # columns: gca, gcf (or similar)

# normalize column names just in case
g = g.rename(columns={"assembly_accession": "GCA"})
a = a.rename(columns={"resistant_phenotype": "Resistance phenotype",
                      "antibiotic": "Antibiotic"})

# join AMR -> genomes to get GCA
df = a.merge(g[["genome_id", "GCA"]], on="genome_id", how="left")

# join GCA -> GCF
# your mapping file produced by gca_to_gcf.py might not be named exactly; adjust if needed
cols = [c.lower() for c in m.columns]
m.columns = cols
if "gca" not in m.columns or "gcf" not in m.columns:
    raise SystemExit(
        f"Expected columns gca,gcf in {MAP}, got {m.columns.tolist()}")

df["GCA"] = df["GCA"].astype(str).str.strip()
m["gca"] = m["gca"].astype(str).str.strip()
df = df.merge(m[["gca", "gcf"]], left_on="GCA", right_on="gca", how="left")

# normalize phenotype
df["p"] = df["Resistance phenotype"].map(norm_pheno)
df = df.dropna(subset=["gcf", "p", "Antibiotic"])

# keep only needed columns
df = df[["gcf", "Antibiotic", "p", "testing_standard",
         "testing_standard_year"]].copy()

df.to_csv(OUT, index=False)
print("[OK] wrote", OUT, "rows=", len(df))
print(df.head(5).to_string(index=False))
