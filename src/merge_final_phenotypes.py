"""
Merge all phenotype data with complete genome set
"""
import pandas as pd
import numpy as np

print("="*80)
print("MERGING ALL PHENOTYPE DATA")
print("="*80)

# Load NCBI phenotypes
ncbi = pd.read_csv("data/amr_tables/phenotypes_clean.csv")
print(f"NCBI phenotypes: {len(ncbi):,}")

# Load BioSample mapping for NCBI
biosample_map = pd.read_csv("data/genomes/genome_to_biosample.csv")
biosample_to_gcf = dict(zip(biosample_map['BioSample'], biosample_map['genome_id']))

# Map NCBI phenotypes
ncbi['gcf_id'] = ncbi['BioSample'].map(lambda bs: biosample_to_gcf.get(bs, ''))
ncbi = ncbi[ncbi['gcf_id'] != '']
ncbi['gcf_norm'] = ncbi['gcf_id'].str.split('.').str[0]

# Clean NCBI
ncbi_clean = ncbi[['gcf_norm', 'Antibiotic', 'Resistance phenotype']].copy()
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

print(f"NCBI cleaned: {len(ncbi_clean):,}")

# Load BV-BRC phenotypes
bvbrc = pd.read_csv("data/bvbrc/kp_amr.csv")
print(f"BV-BRC phenotypes: {len(bvbrc):,}")

# Load BV-BRC genome mapping
bvbrc_genomes = pd.read_csv("data/bvbrc/kp_genomes.csv")
bvbrc_genomes['gcf_norm'] = bvbrc_genomes['assembly_accession'].str.split('.').str[0]

# Load GCA->GCF mapping
gca_gcf = pd.read_csv("data/bvbrc/gca_to_gcf.csv")
gca_gcf['gca_norm'] = gca_gcf['gca'].str.split('.').str[0]
gca_gcf['gcf_norm'] = gca_gcf['gcf'].str.split('.').str[0]
gca_to_gcf_dict = gca_gcf.set_index('gca_norm')['gcf_norm'].to_dict()

# Map BV-BRC genome_id -> GCF
bvbrc_genomes['gca_norm'] = bvbrc_genomes['assembly_accession'].str.split('.').str[0]
bvbrc_genomes['gcf_from_map'] = bvbrc_genomes['gca_norm'].map(gca_to_gcf_dict)
bvbrc_genomes['gcf_final'] = bvbrc_genomes['gcf_from_map'].fillna(bvbrc_genomes['gcf_norm'])

bvbrc_id_to_gcf = bvbrc_genomes.set_index('genome_id')['gcf_final'].to_dict()

# Map BV-BRC phenotypes
bvbrc['gcf_norm'] = bvbrc['genome_id'].map(bvbrc_id_to_gcf)
bvbrc = bvbrc[bvbrc['gcf_norm'].notna()]

bvbrc_clean = bvbrc[['gcf_norm', 'antibiotic', 'resistant_phenotype']].copy()
bvbrc_clean.columns = ['genome_id', 'antibiotic', 'phenotype']
bvbrc_clean['phenotype'] = bvbrc_clean['phenotype'].apply(normalize_phenotype)
bvbrc_clean = bvbrc_clean[bvbrc_clean['phenotype'].notna()]
bvbrc_clean['antibiotic'] = bvbrc_clean['antibiotic'].str.lower().str.strip()

print(f"BV-BRC cleaned: {len(bvbrc_clean):,}")

# Combine
all_pheno = pd.concat([ncbi_clean, bvbrc_clean], ignore_index=True)

# Deduplicate
all_pheno = all_pheno.drop_duplicates(subset=['genome_id', 'antibiotic'])

# Filter to genomes we actually have
with open("data/ml/genome_ids_all_complete.txt") as f:
    available_genomes = set(line.strip().split('.')[0] for line in f)

all_pheno = all_pheno[all_pheno['genome_id'].isin(available_genomes)]

print(f"\n✓ Combined phenotypes: {len(all_pheno):,}")
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
all_pheno.to_csv("data/merged/phenotypes_final.csv", index=False)
print(f"\n✓ Saved: data/merged/phenotypes_final.csv")

print("\n" + "="*80)
print("PIPELINE 1 COMPLETE!")
print("="*80)
print(f"Total unique genomes: 21,621")
print(f"Genomes with phenotypes: {all_pheno['genome_id'].nunique():,}")
print(f"Total phenotype observations: {len(all_pheno):,}")
print("="*80)

