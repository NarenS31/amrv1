import os
import pandas as pd

BV_GENOMES = "data/bvbrc/kp_genomes.csv"
ASM_RS = "data/metadata/assembly_summary_refseq.txt"
OUT = "data/bvbrc/gca_to_gcf.csv"

os.makedirs("data/bvbrc", exist_ok=True)

g = pd.read_csv(BV_GENOMES, low_memory=False)
g = g[g["assembly_accession"].notna()].copy()
g["gca"] = g["assembly_accession"].astype(str).str.strip()
g = g.drop_duplicates(subset=["gca"])

rs = pd.read_csv(ASM_RS, sep="\t", header=1, low_memory=False)
rs = rs.rename(columns={"#assembly_accession": "gcf"})
for c in ["gcf", "gbrs_paired_asm"]:
    rs[c] = rs[c].astype(str).str.strip()

# gbrs_paired_asm is the paired GenBank accession (GCA_...)
x = g.merge(rs[["gcf", "gbrs_paired_asm"]], left_on="gca",
            right_on="gbrs_paired_asm", how="left")

print("[INFO] unique GCA:", len(g))
print("[INFO] mapped to GCF:", int(x["gcf"].notna().sum()))
print("[INFO] unmapped:", int(x["gcf"].isna().sum()))

x[["gca", "gcf"]].to_csv(OUT, index=False)
print("[OK] saved", OUT)
