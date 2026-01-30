"""
Normalize antibiotic names and merge all phenotype data
"""
import pandas as pd
import re

def normalize_antibiotic(name):
    """Normalize antibiotic names for merging"""
    name = str(name).lower().strip()
    # Replace common variations
    name = name.replace('-', '/')
    name = name.replace('_', '/')
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    return name

print("="*80)
print("MERGING PHENOTYPES WITH NORMALIZED NAMES")
print("="*80)

# Load both datasets
old = pd.read_csv("data/merged/all_phenotypes.csv")
new = pd.read_csv("data/merged/all_phenotypes_recovered.csv")

print(f"Old: {len(old):,} observations, {old['genome_id'].nunique():,} genomes")
print(f"New: {len(new):,} observations, {new['genome_id'].nunique():,} genomes")

# Normalize antibiotic names
old['antibiotic_norm'] = old['antibiotic'].apply(normalize_antibiotic)
new['antibiotic_norm'] = new['antibiotic'].apply(normalize_antibiotic)

# Normalize genome IDs (remove version)
old['genome_id_norm'] = old['genome_id'].str.split('.').str[0]
new['genome_id_norm'] = new['genome_id'].str.split('.').str[0]

# Create merge key
old['merge_key'] = old['genome_id_norm'] + '::' + old['antibiotic_norm']
new['merge_key'] = new['genome_id_norm'] + '::' + new['antibiotic_norm']

print(f"\nOld unique combinations: {old['merge_key'].nunique():,}")
print(f"New unique combinations: {new['merge_key'].nunique():,}")

# Find truly new observations
new_only = new[~new['merge_key'].isin(old['merge_key'])].copy()
print(f"Truly new observations: {len(new_only):,}")

if len(new_only) > 0:
    print("\nSample of new observations:")
    print(new_only[['genome_id', 'antibiotic', 'phenotype']].head(10))
    
    # Add new observations to old
    combined = pd.concat([
        old[['genome_id', 'antibiotic', 'phenotype']],
        new_only[['genome_id', 'antibiotic', 'phenotype']]
    ], ignore_index=True)
else:
    combined = old[['genome_id', 'antibiotic', 'phenotype']].copy()

# Final dedup
combined = combined.drop_duplicates()

print(f"\n✓ Final combined dataset: {len(combined):,} observations")
print(f"✓ Unique genomes: {combined['genome_id'].nunique():,}")
print(f"✓ Unique antibiotics: {combined['antibiotic'].nunique()}")

# Show top antibiotics
print("\nTop 20 antibiotics by sample count:")
counts = combined.groupby('antibiotic').size().sort_values(ascending=False)
for ab, count in counts.head(20).items():
    print(f"  {ab:35s}: {count:,}")

# Save
combined.to_csv("data/merged/all_phenotypes_final.csv", index=False)
print(f"\n✓ Saved: data/merged/all_phenotypes_final.csv")

print("\n" + "="*80)
print("IMPROVEMENT")
print("="*80)
print(f"Original: {len(old):,} observations")
print(f"Final:    {len(combined):,} observations")
print(f"Gained:   +{len(combined) - len(old):,} observations (+{(len(combined) - len(old))/len(old)*100:.1f}%)")
print("="*80)

