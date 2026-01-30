import pandas as pd

# Read the raw CSV
phenotypes = pd.read_csv("data/amr_tables/phenotypes_raw.csv")

# Rename the column so it’s easier to work with
phenotypes.rename(columns={"#BioSample": "BioSample"}, inplace=True)

# Keep only the columns we actually care about for ML
essential_cols = [
    "BioSample",
    "Scientific name",
    "Isolate",
    "Antibiotic",
    "Resistance phenotype",
    "MIC (mg/L)"
]

cleaned = phenotypes[essential_cols]

# Save cleaned CSV
cleaned.to_csv("data/amr_tables/phenotypes_clean.csv", index=False)

print(f"Cleaned dataset saved, total rows: {len(cleaned)}")
