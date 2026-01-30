"""
Recover all NCBI + BV-BRC phenotypes using proper mappings
"""
import pandas as pd
import numpy as np

print("="*80)
print("RECOVERING ALL PHENOTYPE DATA")
print("="*80)

# 1. Load BioSample → GCF mapping
biosample_map = pd.read_csv("data/genomes/genome_to_biosample.csv")
print(f"BioSample mappings: {len(biosample_map):,}")

# Normalize
biosample_map['gcf_norm'] = biosample_map['genome_id'].str.split('.').str[0]
biosample_to_gcf = biosample_map.set_index('BioSample')['gcf_norm'].to_dict()

# 2. Load combined genome IDs
with open("data/combined/genome_ids_all.txt") as f:
    combined_gcf = set(line.strip().split('.')[0] for line in f)

print(f"Combined GCF IDs: {len(combined_gcf):,}")

# 3. Load NCBI phenotypes
ncbi = pd.read_csv("data/amr_tables/phenotypes_clean.csv")
print(f"\nNCBI phenotypes: {len(ncbi):,}")

# Map NCBI BioSample → GCF
ncbi['gcf_norm'] = ncbi['BioSample'].map(biosample_to_gcf)
ncbi_matched = ncbi[ncbi['gcf_norm'].isin(combined_gcf)].copy()

print(f"NCBI matched to combined: {len(ncbi_matched):,}")

# Clean NCBI phenotypes
ncbi_clean = ncbi_matched[['gcf_norm', 'Antibiotic', 'Resistance phenotype']].copy()
ncbi_clean.columns = ['genome_id', 'antibiotic', 'phenotype']

def normalize_phenotype(val):
    val = str(val).lower().strip()
    if 'susceptible' in val or val == 's':
        return 'S'
    elif 'resistant' in val or val == 'r':
        return 'R'
    elif 'intermediate' in val or val == 'i':
        return 'I'
    return None

ncbi_clean['phenotype'] = ncbi_clean['phenotype'].apply(normalize_phenotype)
ncbi_clean = ncbi_clean[ncbi_clean['phenotype'].notna()]
ncbi_clean['antibiotic'] = ncbi_clean['antibiotic'].str.lower().str.strip()

print(f"NCBI cleaned: {len(ncbi_clean):,} observations")

# 4. Load BV-BRC data (already recovered)
bvbrc_recovered = pd.read_csv("data/merged/all_phenotypes_recovered.csv")
print(f"BV-BRC recovered: {len(bvbrc_recovered):,}")

# Normalize BV-BRC
bvbrc_recovered['antibiotic'] = bvbrc_recovered['antibiotic'].str.lower().str.strip()
bvbrc_recovered['genome_id'] = bvbrc_recovered['genome_id'].str.split('.').str[0]

# 5. Combine NCBI + BV-BRC
all_pheno = pd.concat([ncbi_clean, bvbrc_recovered], ignore_index=True)

# Deduplicate (genome + antibiotic)
all_pheno = all_pheno.drop_duplicates(subset=['genome_id', 'antibiotic'])

print(f"\n✓ Combined total: {len(all_pheno):,} observations")
print(f"✓ Unique genomes: {all_pheno['genome_id'].nunique():,}")
print(f"✓ Unique antibiotics: {all_pheno['antibiotic'].nunique()}")

# Show top antibiotics
print("\nTop 20 antibiotics:")
counts = all_pheno.groupby('antibiotic').size().sort_values(ascending=False)
for ab, count in counts.head(20).items():
    print(f"  {ab:35s}: {count:,}")

# Observations per genome
obs_per_genome = all_pheno.groupby('genome_id').size()
print(f"\nObservations per genome:")
print(f"  Mean: {obs_per_genome.mean():.1f}")
print(f"  Median: {obs_per_genome.median():.0f}")
print(f"  Genomes with 10+ tests: {(obs_per_genome >= 10).sum():,}")

# Save
all_pheno.to_csv("data/merged/all_phenotypes_complete.csv", index=False)
print(f"\n✓ Saved: data/merged/all_phenotypes_complete.csv")

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"Original merged: 15,618 observations")
print(f"Fully recovered: {len(all_pheno):,} observations")
print(f"Gained: +{len(all_pheno) - 15618:,} (+{(len(all_pheno) - 15618)/15618*100:.1f}%)")
print("="*80)

