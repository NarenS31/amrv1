#!/bin/bash
# Get all genomes that have phenotype labels

echo "Getting genomes with phenotype labels..."

# Get unique genome IDs from phenotype data
python3 << PYEOF
import pandas as pd

pheno = pd.read_csv("data/merged/phenotypes_final.csv")
unique_genomes = pheno['genome_id'].unique()

print(f"Found {len(unique_genomes)} unique genomes with phenotypes")

# Save to file
with open("genomes_with_phenotypes.txt", 'w') as f:
    for gid in unique_genomes:
        f.write(gid + '\n')

print("Saved to genomes_with_phenotypes.txt")
PYEOF

# Create batch file with genome paths
echo "Creating batch file with genome paths..."

> amrfinder_batch_all.txt

while read genome_id; do
    # Find the genome file
    genome_file=$(find data/genomes -name "${genome_id}*.fna" | head -1)
    
    if [ -n "$genome_file" ]; then
        echo "$genome_file" >> amrfinder_batch_all.txt
    fi
done < genomes_with_phenotypes.txt

echo "Total genomes to process: $(wc -l < amrfinder_batch_all.txt)"

