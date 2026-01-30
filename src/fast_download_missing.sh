#!/bin/bash

echo "DOWNLOADING MISSING GENOMES VIA NCBI FTP"
echo "========================================"

# Create directory
mkdir -p data/ncbi_missing_fna

# Read BioSamples and search for assemblies
echo "Searching for assembly accessions..."

python - << 'EOFPYTHON'
import pandas as pd
import requests
import time
from pathlib import Path

# Load missing BioSamples
missing = pd.read_csv("data/missing_biosamples.csv")
biosamples = missing['BioSample'].tolist()[:100]  # Start with first 100

print(f"Processing {len(biosamples)} BioSamples...")

assemblies = []

for i, bs in enumerate(biosamples):
    if i % 10 == 0:
        print(f"  Progress: {i}/{len(biosamples)}")
    
    # Query NCBI
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'assembly',
        'term': f'{bs}[BioSample] AND "Klebsiella pneumoniae"[Organism]',
        'retmode': 'json',
        'retmax': 1
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if 'esearchresult' in data and data['esearchresult']['idlist']:
            assembly_id = data['esearchresult']['idlist'][0]
            
            # Get assembly details
            url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params2 = {
                'db': 'assembly',
                'id': assembly_id,
                'retmode': 'json'
            }
            
            r2 = requests.get(url2, params=params2, timeout=10)
            data2 = r2.json()
            
            if 'result' in data2 and assembly_id in data2['result']:
                acc = data2['result'][assembly_id].get('assemblyaccession')
                if acc:
                    assemblies.append({'BioSample': bs, 'Assembly': acc})
                    print(f"    Found: {bs} -> {acc}")
        
        time.sleep(0.35)  # Rate limit
        
    except Exception as e:
        print(f"    Error for {bs}: {e}")
        continue

# Save
df = pd.DataFrame(assemblies)
df.to_csv("data/ncbi_missing/biosample_to_assembly.csv", index=False)
print(f"\n✓ Found {len(assemblies)} assemblies")
print(f"✓ Saved to data/ncbi_missing/biosample_to_assembly.csv")

EOFPYTHON

echo "Done!"

