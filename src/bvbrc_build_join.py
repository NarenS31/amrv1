import pandas as pd

GENOMES = "data/bvbrc/kp_genomes.csv"
AMR = "data/bvbrc/kp_amr.csv"
MAP = "data/bvbrc/gca_to_gcf.csv"
OUT = "data/bvbrc/bvbrc_joined.csv"

g = pd.read_csv(GENOMES, low_memory=False)
a = pd.read_csv(AMR, low_memory=False)
m = pd.read_csv(MAP, low_memory=False)   # cols: gca, gcf (or similar)

# --- normalize column names ---
g = g.rename(columns={"assembly_accession": "assembly"})
a = a.rename(columns={"resistant_phenotype": "phenotype"})

# keep minimal
g = g[["genome_id", "assembly"]].dropna()
a = a[["genome_id", "antibiotic", "phenotype"]].dropna()

# map GCA->GCF when possible
# adjust column names if your csv uses different headers
if "gca" in m.columns and "gcf" in m.columns:
    pass
else:
    # fallback guess
    m.columns = ["gca", "gcf"]

g = g.merge(m, left_on="assembly", right_on="gca", how="left")
# if already GCF or unmapped keep original
g["assembly_gcf"] = g["gcf"].fillna(g["assembly"])
g = g.drop(columns=[c for c in ["gca", "gcf"] if c in g.columns])

# join AMR to assembly_gcf
j = a.merge(g[["genome_id", "assembly_gcf"]], on="genome_id", how="inner")

# normalize phenotype to S/I/R (BV-BRC may have strings)


def norm(x):
    x = str(x).strip().upper()
    if x in ["S", "SUSCEPTIBLE"]:
        return "S"
    if x in ["I", "INTERMEDIATE"]:
        return "I"
    if x in ["R", "RESISTANT"]:
        return "R"
    return None


j["p"] = j["phenotype"].map(norm)
j = j.dropna(subset=["p"])

# save
j = j.rename(columns={"assembly_gcf": "assembly"})
j.to_csv(OUT, index=False)

print("[OK] wrote", OUT)
print("rows:", len(j), "unique assemblies:", j["assembly"].nunique(
), "unique antibiotics:", j["antibiotic"].nunique())
print("p counts:\n", j["p"].value_counts())
