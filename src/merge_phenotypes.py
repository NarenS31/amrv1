
import os
import pandas as pd
import numpy as np


def normalize_phenotype(val):
    """Map to S/I/R"""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()

    if s in {"s", "susceptible"}:
        return "S"
    if s in {"r", "resistant"}:
        return "R"
    if s in {"i", "intermediate"}:
        return "I"

    if "suscept" in s:
        return "S"
    if "intermed" in s:
        return "I"
    if "resist" in s and "non" not in s:
        return "R"

    return None


def normalize_id(gid):
    """Normalize genome IDs: GCF_000123456.1 -> GCF_000123456"""
    if pd.isna(gid):
        return None
    s = str(gid).strip()

    if s.startswith("GCF_") or s.startswith("GCA_"):
        s = s.split(".")[0]
        parts = s.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return s


def resolve_conflict(group):
    """R > S > I"""
    vals = set(group["phenotype"].dropna().values)
    if "R" in vals:
        return "R"
    if "S" in vals:
        return "S"
    if "I" in vals:
        return "I"
    return group["phenotype"].iloc[0]


print("="*80)
print("MERGING PHENOTYPE DATA")
print("="*80)

# ============================================================================
# LOAD ORIGINAL PHENOTYPES
# ============================================================================
print("\n[1/3] Loading original phenotypes...")

orig_pheno_raw = pd.read_csv(
    "data/amr_tables/phenotypes_clean.csv", low_memory=False)
print(f"  Original rows: {len(orig_pheno_raw)}")
print(f"  Columns: {orig_pheno_raw.columns.tolist()}")

# Load BioSample -> genome_id mapping
genome_map = pd.read_csv("data/genomes/genome_to_biosample.csv")
print(f"  Loaded genome mapping: {len(genome_map)} rows")

# Merge to get genome_id
orig_pheno_raw = orig_pheno_raw.merge(
    genome_map[['BioSample', 'genome_id']],
    on='BioSample',
    how='left'
)

print(
    f"  After merge: {orig_pheno_raw['genome_id'].notna().sum()} rows with genome_id")

# Rename columns to standard format
orig_pheno = orig_pheno_raw.rename(columns={
    'Antibiotic': 'antibiotic',
    'Resistance phenotype': 'phenotype'
})[['genome_id', 'antibiotic', 'phenotype']].copy()

# Clean
orig_pheno['genome_id'] = orig_pheno['genome_id'].apply(normalize_id)
orig_pheno['phenotype'] = orig_pheno['phenotype'].apply(normalize_phenotype)
orig_pheno['antibiotic'] = orig_pheno['antibiotic'].astype(
    str).str.strip().str.lower()

orig_pheno = orig_pheno[
    orig_pheno['genome_id'].notna() &
    orig_pheno['antibiotic'].notna() &
    orig_pheno['phenotype'].notna()
]

print(f"  Original cleaned rows: {len(orig_pheno)}")

# ============================================================================
# LOAD BV-BRC PHENOTYPES
# ============================================================================
print("\n[2/3] Loading BV-BRC phenotypes...")

bvbrc_amr = pd.read_csv("data/bvbrc/kp_amr.csv", low_memory=False)
bvbrc_genomes = pd.read_csv("data/bvbrc/kp_genomes.csv", low_memory=False)

print(f"  BV-BRC AMR rows: {len(bvbrc_amr)}")

# Load GCA->GCF mapping
if os.path.exists("data/bvbrc/gca_to_gcf.csv"):
    gca_to_gcf = pd.read_csv("data/bvbrc/gca_to_gcf.csv")
    print("  Loaded GCA->GCF mapping")
else:
    gca_to_gcf = pd.DataFrame({'gca': [], 'gcf': []})
    print("  Warning: No GCA->GCF mapping")

# Map BV-BRC genome_id -> GCA -> GCF
genome_to_gca = dict(
    zip(bvbrc_genomes['genome_id'], bvbrc_genomes['assembly_accession']))
gca_to_gcf_dict = dict(zip(gca_to_gcf['gca'], gca_to_gcf['gcf'])) if len(
    gca_to_gcf) > 0 else {}

bvbrc_amr['gca'] = bvbrc_amr['genome_id'].map(genome_to_gca)
bvbrc_amr['gcf'] = bvbrc_amr['gca'].map(
    gca_to_gcf_dict) if gca_to_gcf_dict else bvbrc_amr['gca']

# Normalize
bvbrc_amr['phenotype'] = bvbrc_amr['resistant_phenotype'].apply(
    normalize_phenotype)
bvbrc_amr['genome_id_norm'] = bvbrc_amr['gcf'].apply(normalize_id)
bvbrc_amr['antibiotic'] = bvbrc_amr['antibiotic'].astype(
    str).str.strip().str.lower()

bvbrc_amr_clean = bvbrc_amr[
    bvbrc_amr['phenotype'].notna() &
    bvbrc_amr['genome_id_norm'].notna()
][['genome_id_norm', 'antibiotic', 'phenotype']].copy()

bvbrc_amr_clean = bvbrc_amr_clean.rename(
    columns={'genome_id_norm': 'genome_id'})

print(f"  BV-BRC clean rows: {len(bvbrc_amr_clean)}")

# ============================================================================
# COMBINE AND RESOLVE
# ============================================================================
print("\n[3/3] Combining and resolving conflicts...")

combined = pd.concat([orig_pheno, bvbrc_amr_clean], ignore_index=True)
print(f"  Combined rows: {len(combined)}")

resolved = combined.groupby(['genome_id', 'antibiotic']).apply(
    lambda g: pd.Series({'phenotype': resolve_conflict(g)}),
    include_groups=False
).reset_index()

print(f"  Resolved rows: {len(resolved)}")
print(f"  Unique genomes: {resolved['genome_id'].nunique()}")
print(f"  Unique antibiotics: {resolved['antibiotic'].nunique()}")

# Save
os.makedirs("data/merged", exist_ok=True)
resolved.to_csv("data/merged/all_phenotypes.csv", index=False)

print(f"\n✓ Saved: data/merged/all_phenotypes.csv")

# Show top antibiotics
print("\n" + "="*80)
print("TOP 20 ANTIBIOTICS BY SAMPLE COUNT")
print("="*80)

top_ab = resolved.groupby('antibiotic').size(
).sort_values(ascending=False).head(20)

for ab, count in top_ab.items():
    pheno_counts = resolved[resolved['antibiotic']
                            == ab]['phenotype'].value_counts()
    s = pheno_counts.get('S', 0)
    i = pheno_counts.get('I', 0)
    r = pheno_counts.get('R', 0)
    print(f"{ab:35s}: n={count:5d}  S={s:5d}  I={i:4d}  R={r:5d}")

print("="*80)
