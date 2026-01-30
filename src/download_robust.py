"""
Robust genome download with retry and error handling
"""
import subprocess
import pandas as pd
import time
from pathlib import Path
import os

print("="*80)
print("ROBUST GENOME DOWNLOAD")
print("="*80)

# Load assemblies
assemblies = pd.read_csv("data/ncbi_missing/biosample_to_assembly.csv")
accessions = assemblies['Assembly'].tolist()

print(f"Total genomes to download: {len(accessions):,}")

# Create output directory
output_dir = Path("data/ncbi_missing_fna")
output_dir.mkdir(exist_ok=True)

# Download one at a time with retry
successful = 0
failed = []

for i, acc in enumerate(accessions):
    if i % 10 == 0:
        print(f"\nProgress: {i}/{len(accessions)} (Success: {successful}, Failed: {len(failed)})")
    
    # Check if already downloaded
    fna_file = output_dir / f"{acc}.fna"
    if fna_file.exists() and fna_file.stat().st_size > 100000:  # >100KB
        successful += 1
        continue
    
    # Try to download
    max_retries = 3
    success = False
    
    for retry in range(max_retries):
        try:
            # Download
            cmd = f"datasets download genome accession {acc} --filename temp_{acc}.zip"
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(f"temp_{acc}.zip"):
                # Extract
                extract_cmd = f"unzip -q -o temp_{acc}.zip -d temp_{acc}_data"
                subprocess.run(extract_cmd, shell=True, capture_output=True)
                
                # Find and move fna file
                fna_files = list(Path(f"temp_{acc}_data").rglob("*.fna"))
                if fna_files:
                    fna_files[0].rename(fna_file)
                    successful += 1
                    success = True
                    print(f"  ✓ {acc}")
                    break
                
            # Clean up
            if os.path.exists(f"temp_{acc}.zip"):
                os.remove(f"temp_{acc}.zip")
            if os.path.exists(f"temp_{acc}_data"):
                subprocess.run(f"rm -rf temp_{acc}_data", shell=True)
            
            if not success and retry < max_retries - 1:
                time.sleep(2)  # Wait before retry
                
        except Exception as e:
            print(f"  ✗ {acc}: {e}")
            continue
    
    if not success:
        failed.append(acc)
        print(f"  ✗ {acc} (gave up after {max_retries} retries)")
    
    # Rate limit
    time.sleep(0.5)

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
print(f"Successful: {successful:,}")
print(f"Failed: {len(failed):,}")

if failed:
    pd.DataFrame({'Assembly': failed}).to_csv("data/ncbi_missing/download_failed.csv", index=False)
    print(f"✓ Saved failed list to data/ncbi_missing/download_failed.csv")

print(f"\nGenome files: {len(list(output_dir.glob('*.fna'))):,}")

