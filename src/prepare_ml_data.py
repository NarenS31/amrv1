# # import numpy as np
# # import pandas as pd
# # import numpy as np
# # import pandas as pd

# # # Load k-mer matrix and genome IDs
# # X = np.load("data/genomes/kmer_matrix.npy")
# # with open("data/genomes/genome_ids.txt") as f:
# #     genome_ids = f.read().splitlines()

# # # Shorten genome IDs to match mapping
# # # mapping genome_ids: GCF_000009885.1_ASM988v1 -> GCF_000009885.1
# # short_genome_ids = ["_".join(gid.split("_")[:2]) for gid in genome_ids]

# # # Load genome -> BioSample mapping
# # mapping = pd.read_csv("data/genomes/genome_to_biosample.csv")

# # # Load phenotype table
# # phenos = pd.read_csv("data/amr_tables/phenotypes_clean.csv")

# # # Merge mapping with phenotypes on BioSample
# # merged = mapping.merge(phenos, left_on="BioSample", right_on="BioSample")
# # print("Merged rows:", merged.shape[0])

# # # Find indices of genomes in k-mer matrix
# # indices = []
# # missing = 0
# # for gid in merged['genome_id']:
# #     try:
# #         idx = short_genome_ids.index(gid)
# #         indices.append(idx)
# #     except ValueError:
# #         missing += 1
# # print(f"Genomes not found in k-mer matrix: {missing}")

# # # Extract features and labels
# # X_train = X[indices, :]
# # y_train = merged['Antibiotic'].values

# # print("X_train shape:", X_train.shape)
# # print("y_train shape:", y_train.shape)


# # # after X_train, y_train are created
# # np.save("data/genomes/X_train.npy", X_train)
# # np.save("data/genomes/y_train.npy", y_train)
# # print("Saved X_train and y_train to disk")


# # # Load k-mer matrix and genome IDs
# # X = np.load("data/genomes/kmer_matrix.npy")
# # with open("data/genomes/genome_ids.txt") as f:
# #     genome_ids = f.read().splitlines()

# # # Shorten genome IDs to match mapping
# # # mapping genome_ids: GCF_000009885.1_ASM988v1 -> GCF_000009885.1
# # short_genome_ids = ["_".join(gid.split("_")[:2]) for gid in genome_ids]

# # # Load genome -> BioSample mapping
# # mapping = pd.read_csv("data/genomes/genome_to_biosample.csv")

# # # Load phenotype table
# # phenos = pd.read_csv("data/amr_tables/phenotypes_clean.csv")

# # # Merge mapping with phenotypes on BioSample
# # merged = mapping.merge(phenos, left_on="BioSample", right_on="BioSample")
# # print("Merged rows:", merged.shape[0])

# # # Find indices of genomes in k-mer matrix
# # indices = []
# # missing = 0
# # for gid in merged['genome_id']:
# #     try:
# #         idx = short_genome_ids.index(gid)
# #         indices.append(idx)
# #     except ValueError:
# #         missing += 1
# # print(f"Genomes not found in k-mer matrix: {missing}")

# # # Extract features and labels
# # X_train = X[indices, :]
# # y_train = merged['Antibiotic'].values

# # print("X_train shape:", X_train.shape)
# # print("y_train shape:", y_train.shape)


# # # after X_train, y_train are created
# # np.save("data/genomes/X_train.npy", X_train)
# # np.save("data/genomes/y_train.npy", y_train)
# # print("Saved X_train and y_train to disk")

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# # ===============================
# # 1. LOAD K-MER MATRIX + GENOME IDS
# # ===============================
# X = np.load("data/genomes/kmer_matrix.npy")

# with open("data/genomes/genome_ids.txt") as f:
#     genome_ids = f.read().splitlines()

# # Convert:
# # GCF_000009885.1_ASM988v1 -> GCF_000009885.1
# short_genome_ids = ["_".join(gid.split("_")[:2]) for gid in genome_ids]

# print("K-mer matrix shape:", X.shape)
# print("Number of genome IDs:", len(short_genome_ids))


# # ===============================
# # 2. LOAD MAPPING + PHENOTYPES
# # ===============================
# mapping = pd.read_csv("data/genomes/genome_to_biosample.csv")
# phenos = pd.read_csv("data/amr_tables/phenotypes_clean.csv")

# print("Mapping columns:", mapping.columns.tolist())
# print("Phenotype columns:", phenos.columns.tolist())


# # ===============================
# # 3. MERGE ON BioSample
# # ===============================
# merged = mapping.merge(
#     phenos,
#     left_on="BioSample",
#     right_on="BioSample",
#     how="inner"
# )

# print("Merged rows:", merged.shape[0])


# # ===============================
# # 4. FIND GENOME INDICES
# # ===============================
# indices = []
# missing = 0

# for gid in merged["genome_id"]:
#     try:
#         idx = short_genome_ids.index(gid)
#         indices.append(idx)
#     except ValueError:
#         missing += 1

# print("Genomes not found in k-mer matrix:", missing)


# # ===============================
# # 5. BUILD X AND y
# # ===============================
# X_train = X[indices, :]

# # Encode Antibiotic labels → integers
# le = LabelEncoder()
# y_train = le.fit_transform(merged["Antibiotic"])

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("Number of classes:", len(le.classes_))


# # ===============================
# # 6. SAVE OUTPUTS
# # ===============================
# np.save("data/genomes/X_train.npy", X_train)
# np.save("data/genomes/y_train.npy", y_train)
# np.save("data/genomes/label_classes.npy", le.classes_)

# print("Saved:")
# print("- data/genomes/X_train.npy")
# print("- data/genomes/y_train.npy")
# print("- data/genomes/label_classes.npy")

# src/prepare_ml_data.py
import pandas as pd
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
KMER_MATRIX_PATH = "data/genomes/kmer_matrix.npy"
GENOME_IDS_PATH = "data/genomes/genome_ids.txt"
GENOME_TO_BIOSAMPLE_PATH = "data/genomes/genome_to_biosample.csv"
ASSEMBLY_CROSSWALK_PATH = "data/genomes/assembly_to_biosample.csv"
PHENOTYPES_PATH = "data/amr_tables/phenotypes_raw.csv"

OUT_X = "data/genomes/X_train.npy"
OUT_Y = "data/genomes/y_train.npy"                 # 0/1/2 (S/I/R)
OUT_AB_INDEX = "data/genomes/ab_index.npy"         # antibiotic id per row
OUT_LABELS = "data/genomes/label_classes.npy"      # antibiotic names

# -----------------------------
# Load k-mer data
# -----------------------------
X_full = np.load(KMER_MATRIX_PATH)
genome_ids = pd.read_csv(GENOME_IDS_PATH, header=None, names=["genome_id"])

if len(genome_ids) != X_full.shape[0]:
    raise ValueError(
        f"Genome ID count mismatch: genome_ids={len(genome_ids)} vs X_full={X_full.shape[0]}"
    )

# Extract assembly accession from genome_ids too (for robust joining)
genome_ids["assembly"] = genome_ids["genome_id"].astype(str).str.extract(
    r"(GCA_\d+\.\d+|GCF_\d+\.\d+)", expand=False
)

# -----------------------------
# Load genome → biosample mapping
# -----------------------------
mapping = pd.read_csv(GENOME_TO_BIOSAMPLE_PATH)

# Ensure expected columns exist
if "genome_id" not in mapping.columns:
    raise ValueError(
        f"{GENOME_TO_BIOSAMPLE_PATH} missing genome_id. Columns: {mapping.columns.tolist()}")
if "BioSample" not in mapping.columns:
    raise ValueError(
        f"{GENOME_TO_BIOSAMPLE_PATH} missing BioSample. Columns: {mapping.columns.tolist()}")

# Extract assembly accession from mapping genome_id
mapping["assembly"] = mapping["genome_id"].astype(str).str.extract(
    r"(GCA_\d+\.\d+|GCF_\d+\.\d+)", expand=False
)

# -----------------------------
# Resolve SAMN via assembly crosswalk
# -----------------------------
cross = pd.read_csv(ASSEMBLY_CROSSWALK_PATH)

# Expected columns: assembly, biosample_samn
if "assembly" not in cross.columns or "biosample_samn" not in cross.columns:
    raise ValueError(
        f"{ASSEMBLY_CROSSWALK_PATH} missing columns. Columns: {cross.columns.tolist()}")

# Merge on assembly (NOT genome_id)
mapping = mapping.merge(
    cross[["assembly", "biosample_samn"]],
    on="assembly",
    how="left"
)

# Prefer biosample_samn when present
mapping["BioSample"] = mapping["biosample_samn"].fillna(mapping["BioSample"])
mapping = mapping.drop(columns=["biosample_samn"])

# Normalize mapping BioSample
mapping["BioSample"] = mapping["BioSample"].astype(str).str.strip().str.upper()

# De-duplicate any accidental duplicate columns
mapping = mapping.loc[:, ~mapping.columns.duplicated()]

# -----------------------------
# Load phenotypes
# -----------------------------
ph = pd.read_csv(PHENOTYPES_PATH)

# Sanity checks
needed = ["#BioSample", "Antibiotic", "Resistance phenotype"]
missing = [c for c in needed if c not in ph.columns]
if missing:
    raise ValueError(
        f"{PHENOTYPES_PATH} missing columns: {missing}. Columns: {ph.columns.tolist()}")

# Create join key
ph["BioSample"] = ph["#BioSample"].astype(str).str.strip().str.upper()

# Rename only what we use
ph = ph.rename(columns={
    "Antibiotic": "antibiotic",
    "Resistance phenotype": "phenotype"
})

# Keep only needed columns
ph = ph[["BioSample", "antibiotic", "phenotype"]].copy()

# Normalize phenotype strings
ph["phenotype"] = ph["phenotype"].astype(str).str.strip().str.lower()

# Map to 3-class: S=0, I=1, R=2
map3 = {
    "susceptible": 0, "sensitive": 0, "s": 0,
    "intermediate": 1, "i": 1,
    "resistant": 2, "r": 2
}
ph["label3"] = ph["phenotype"].map(map3)

# Keep valid S/I/R only
ph = ph.dropna(subset=["label3"]).copy()
ph["label3"] = ph["label3"].astype(np.int8)

# -----------------------------
# Debug join health
# -----------------------------
print("\n=== DEBUG BIOSAMPLE CHECK ===")
print("mapping BioSample examples:",
      mapping["BioSample"].dropna().unique()[:10])
print("ph BioSample examples:", ph["BioSample"].dropna().unique()[:10])
print("mapping BioSample count:", mapping["BioSample"].nunique())
print("ph BioSample count:", ph["BioSample"].nunique())

overlap = set(mapping["BioSample"]) & set(ph["BioSample"])
print("OVERLAP SIZE:", len(overlap))
print("overlap examples:", list(overlap)[:10])
print("============================\n")

# -----------------------------
# Merge genomes ↔ phenotypes
# -----------------------------
merged = mapping.merge(ph, on="BioSample", how="inner")
print("Merged rows:", len(merged))

if len(merged) == 0:
    raise RuntimeError("Merged rows is 0. Your BioSample join has no overlap.")

# -----------------------------
# Build feature + label vectors
# Each row = one (genome, antibiotic measurement)
# X row = genome k-mer vector
# y row = 0/1/2 (S/I/R)
# ab_index row = antibiotic id
# -----------------------------
asm_to_idx = {
    a: i for i, a in enumerate(genome_ids["assembly"])
    if isinstance(a, str) and a != "" and a != "nan"
}

merged["row_idx"] = merged["assembly"].map(asm_to_idx)
mapped = merged["row_idx"].notna().sum()
print(f"[DEBUG] row_idx mapped: {mapped} / {len(merged)}")

if mapped == 0:
    raise RuntimeError(
        "row_idx mapping is 0. genome_ids.txt assembly extraction didn't match mapping assemblies."
    )

merged = merged.dropna(subset=["row_idx"]).copy()
merged["row_idx"] = merged["row_idx"].astype(int)

# Antibiotic vocabulary
labels = sorted(merged["antibiotic"].dropna().unique())
label_to_idx = {l: i for i, l in enumerate(labels)}

X = np.zeros((len(merged), X_full.shape[1]), dtype=np.float32)
y = np.zeros((len(merged),), dtype=np.int8)          # 0/1/2
ab_index = np.zeros((len(merged),), dtype=np.int32)

for i, (_, r) in enumerate(merged.iterrows()):
    X[i] = X_full[r["row_idx"]]
    y[i] = r["label3"]
    ab_index[i] = label_to_idx[r["antibiotic"]]

# -----------------------------
# Save
# -----------------------------
os.makedirs("data/genomes", exist_ok=True)

np.save(OUT_X, X)
np.save(OUT_Y, y)
np.save(OUT_AB_INDEX, ab_index)
np.save(OUT_LABELS, np.array(labels, dtype=object))

print("Saved:")
print(f"  {OUT_X} shape={X.shape}")
print(f"  {OUT_Y} shape={y.shape} (0=S,1=I,2=R)")
print(f"  {OUT_AB_INDEX} shape={ab_index.shape}")
print(f"  {OUT_LABELS} labels={len(labels)}")
