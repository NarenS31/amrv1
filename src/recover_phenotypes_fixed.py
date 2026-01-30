import pandas as pd

print("="*80)
print("COMBINING OLD + RECOVERED PHENOTYPES")
print("="*80)

# Load both
old = pd.read_csv("data/merged/all_phenotypes.csv")
new = pd.read_csv("data/merged/all_phenotypes_recovered.csv")

print(
    f"Old phenotypes: {len(old):,} observations, {old['genome_id'].nunique():,} genomes")
print(
    f"Recovered phenotypes: {len(new):,} observations, {new['genome_id'].nunique():,} genomes")

# Combine and deduplicate
combined = pd.concat([old, new], ignore_index=True)
combined = combined.drop_duplicates(subset=['genome_id', 'antibiotic'])

print(
    f"\nCombined: {len(combined):,} observations, {combined['genome_id'].nunique():,} genomes")
print(f"Unique antibiotics: {combined['antibiotic'].nunique()}")

# Show per-antibiotic counts
print("\nTop 20 antibiotics:")
counts = combined.groupby('antibiotic').size().sort_values(ascending=False)
for ab, count in counts.head(20).items():
    print(f"  {ab:30s}: {count:,}")

# Save
combined.to_csv("data/merged/all_phenotypes_combined.csv", index=False)
print(f"\n✓ Saved: data/merged/all_phenotypes_combined.csv")

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"Total observations: {len(combined):,}")
print(f"Total genomes: {combined['genome_id'].nunique():,}")
print(
    f"Improvement: +{len(combined) - len(old):,} observations (+{(len(combined) - len(old))/len(old)*100:.1f}%)")
print("="*80)
