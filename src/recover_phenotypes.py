"""
Re-merge phenotypes using BV-BRC genome ID mapping
"""
import pandas as pd

print("="*80)
print("RECOVERING MISSING PHENOTYPES")
print("="*80)

# Load BV-BRC phenotypes
amr = pd.read_csv("data/bvbrc/kp_amr.csv")
print(f"Original AMR data: {len(amr):,} rows")

# Load BV-BRC genome metadata
genomes = pd.read_csv("data/bvbrc/kp_genomes.csv")
print(f"Genome metadata: {len(genomes):,} rows")

# Load your combined genome IDs
with open("data/combined/genome_ids_all.txt") as f:
    combined_ids = set(line.strip() for line in f)

print(f"Combined genome IDs: {len(combined_ids):,}")

# Normalize GCA to GCF
def gca_to_gcf(acc):
    if pd.isna(acc):
        return None
    return acc.replace('GCA_', 'GCF_').split('.')[0]

# Create mapping: BV-BRC genome_id -> GCF accession
genomes['gcf_norm'] = genomes['assembly_accession'].apply(gca_to_gcf)
bvbrc_to_gcf = genomes.set_index('genome_id')['gcf_norm'].to_dict()

print(f"\nBV-BRC to GCF mappings: {len(bvbrc_to_gcf):,}")

# Map phenotypes to GCF IDs
amr['gcf_id'] = amr['genome_id'].map(bvbrc_to_gcf)

# Keep only genomes in our combined set
amr_matched = amr[amr['gcf_id'].isin(combined_ids)].copy()

print(f"\nPhenotypes matched to combined genomes: {len(amr_matched):,}")
print(f"Unique genomes: {amr_matched['gcf_id'].nunique():,}")

# Extract clean phenotype data
amr_clean = amr_matched[['gcf_id', 'antibiotic', 'resistant_phenotype']].copy()
amr_clean.columns = ['genome_id', 'antibiotic', 'phenotype']

# Normalize phenotype values
def normalize_phenotype(val):
    val = str(val).strip().upper()
    if val in ['SUSCEPTIBLE', 'S']:
        return 'S'
    elif val in ['RESISTANT', 'R']:
        return 'R'
    elif val in ['INTERMEDIATE', 'I']:
        return 'I'
    else:
        return None

amr_clean['phenotype'] = amr_clean['phenotype'].apply(normalize_phenotype)
amr_clean = amr_clean[amr_clean['phenotype'].notna()].drop_duplicates()

print(f"\nAfter cleaning: {len(amr_clean):,} observations")
print(f"Unique genomes: {amr_clean['genome_id'].nunique():,}")
print(f"Unique antibiotics: {amr_clean['antibiotic'].nunique()}")

# Save
amr_clean.to_csv("data/merged/all_phenotypes_recovered.csv", index=False)
print(f"\n✓ Saved: data/merged/all_phenotypes_recovered.csv")

# Show improvement
old = pd.read_csv("data/merged/all_phenotypes.csv")
print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Old: {len(old):,} observations, {old['genome_id'].nunique():,} genomes")
print(f"New: {len(amr_clean):,} observations, {amr_clean['genome_id'].nunique():,} genomes")
print(f"Gained: +{len(amr_clean) - len(old):,} observations (+{(len(amr_clean) - len(old))/len(old)*100:.1f}%)")
print("="*80)

