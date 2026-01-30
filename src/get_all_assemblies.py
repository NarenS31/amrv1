import pandas as pd
import requests
import time
from pathlib import Path

print("="*80)
print("FINDING ASSEMBLIES FOR ALL 2,699 BIOSAMPLES")
print("="*80)

# Create directory
Path("data/ncbi_missing").mkdir(exist_ok=True)

# Load missing BioSamples
missing = pd.read_csv("data/missing_biosamples.csv")
biosamples = missing['BioSample'].tolist()

print(f"Total BioSamples: {len(biosamples):,}")

assemblies = []
failed = []

for i, bs in enumerate(biosamples):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(biosamples)} ({i/len(biosamples)*100:.1f}%)")
    
    try:
        # Search for assembly
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'assembly',
            'term': f'{bs}[BioSample] AND "Klebsiella pneumoniae"[Organism]',
            'retmode': 'json',
            'retmax': 1
        }
        
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if 'esearchresult' in data and data['esearchresult']['idlist']:
            assembly_id = data['esearchresult']['idlist'][0]
            
            # Get details
            url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params2 = {'db': 'assembly', 'id': assembly_id, 'retmode': 'json'}
            
            r2 = requests.get(url2, params=params2, timeout=10)
            data2 = r2.json()
            
            if 'result' in data2 and assembly_id in data2['result']:
                acc = data2['result'][assembly_id].get('assemblyaccession')
                if acc:
                    assemblies.append({'BioSample': bs, 'Assembly': acc})
        else:
            failed.append(bs)
        
        time.sleep(0.35)  # Rate limit
        
    except Exception as e:
        print(f"  Error for {bs}: {e}")
        failed.append(bs)
        continue

# Save results
df = pd.DataFrame(assemblies)
df.to_csv("data/ncbi_missing/biosample_to_assembly.csv", index=False)

print(f"\n✓ Found {len(assemblies):,} assemblies")
print(f"✗ Failed: {len(failed)}")
print(f"✓ Saved to data/ncbi_missing/biosample_to_assembly.csv")

# Save failed for retry
if failed:
    pd.DataFrame({'BioSample': failed}).to_csv("data/ncbi_missing/failed_biosamples.csv", index=False)
    print(f"✓ Saved {len(failed)} failed to data/ncbi_missing/failed_biosamples.csv")

print("\nThis found assembly accessions. Next: Download the genomes.")

