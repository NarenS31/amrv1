import os
import pandas as pd

MAP_PATH = "data/bvbrc/gca_to_gcf.csv"
ASM_RS = "data/metadata/assembly_summary_refseq.txt"
OUT = "data/bvbrc/refseq_download_urls.tsv"

os.makedirs("data/bvbrc", exist_ok=True)
os.makedirs("data/bvbrc/refseq_fna", exist_ok=True)

m = pd.read_csv(MAP_PATH)
m = m[m["gcf"].notna()].copy()
m["gcf"] = m["gcf"].astype(str).str.strip()

# IMPORTANT: dedupe so 10 identical assemblies don't get downloaded 10 times
m = m.drop_duplicates(subset=["gcf"])

rs = pd.read_csv(ASM_RS, sep="\t", header=1, low_memory=False)
rs = rs.rename(columns={"#assembly_accession": "gcf"})
rs["gcf"] = rs["gcf"].astype(str).str.strip()

rs = rs[["gcf", "ftp_path"]].copy()

j = m.merge(rs, on="gcf", how="left")

print("[INFO] unique GCF requested:", len(m))
print("[INFO] ftp found:", int(j["ftp_path"].notna().sum()))
print("[INFO] ftp missing:", int(j["ftp_path"].isna().sum()))


def url_from_ftp(ftp: str | None):
    if not isinstance(ftp, str) or not ftp:
        return None
    base = ftp.rstrip("/").split("/")[-1]
    return f"{ftp}/{base}_genomic.fna.gz"


j["url"] = j["ftp_path"].apply(url_from_ftp)
j = j[j["url"].notna()].copy()

# write as: GCF<TAB>URL
j[["gcf", "url"]].to_csv(OUT, sep="\t", index=False, header=False)
print("[OK] wrote", len(j), "urls ->", OUT)
