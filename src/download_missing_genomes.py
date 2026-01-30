"""
Download the 2,699 missing genomes with phenotype data
"""
import pandas as pd
import requests
from pathlib import Path
import time

print("="*80)
print("DOWNLOADING MISSING GENOMES")
print("="*80)

# Load missing BioSamples
missing = pd.read_csv("data/missing_biosamples.csv")
print(f"Total BioSamples to download: {len(missing):,}")

# Create output directory
output_dir = Path("data/ncbi_missing_genomes")
output_dir.mkdir(exist_ok=True)

# Query NCBI to get assembly accessions for these BioSamples
print("\nQuerying NCBI for assembly accessions...")

biosample_to_assembly = {}
batch_size = 200

for i in range(0, len(missing), batch_size):
    batch = missing['BioSample'].iloc[i:i+batch_size].tolist()
    
    # Search NCBI Assembly database
    search_term = " OR ".join([f"{bs}[BioSample]" for bs in batch])
    
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'assembly',
        'term': search_term,
        'retmax': batch_size,
        'retmode': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'esearchresult' in data and 'idlist' in data['esearchresult']:
            assembly_ids = data['esearchresult']['idlist']
            print(f"  Batch {i//batch_size + 1}: Found {len(assembly_ids)} assemblies")
        
        time.sleep(0.4)  # Rate limit
        
    except Exception as e:
        print(f"  Error in batch {i//batch_size + 1}: {e}")
        continue

print(f"\n✓ Query complete")
print(f"\nNext: We need to fetch assembly details and download URLs")
print(f"This will take approximately 2-3 hours...")

# Save progress
print("\nThis is a BIG download. Recommend running in background.")
print("Script created but not executing full download yet.")
print("\nTo proceed, you would need to:")
print("1. Fetch assembly accessions for all BioSamples")
print("2. Download ~2,700 genome files (10-20 GB)")
print("3. Extract and process k-mers")

