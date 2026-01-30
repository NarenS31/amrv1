import pandas as pd
phenotypes = pd.read_csv("data/amr_tables/phenotypes_clean.csv")

all_antibiotics = phenotypes['Antibiotic'].unique()

print(f"Found {len(all_antibiotics)} antibiotics:")
for ab in all_antibiotics:
    print(ab)
