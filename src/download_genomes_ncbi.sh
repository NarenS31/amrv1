#!/bin/bash

echo "================================"
echo "DOWNLOADING 1,277 GENOMES"
echo "================================"

# Create directories
mkdir -p data/ncbi_missing_fna
mkdir -p data/ncbi_missing_urls

# Read assembly accessions
python - << 'EOFPY'
import pandas as pd

assemblies = pd.read_csv("data/ncbi_missing/biosample_to_assembly.csv")

# Generate NCBI FTP URLs for each assembly
urls = []
for _, row in assemblies.iterrows():
    acc = row['Assembly']
    
    # Remove version
    acc_base = acc.split('.')[0]
    
    # GCF or GCA
    if acc.startswith('GCF'):
        base_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/{acc_base[4:7]}/{acc_base[7:10]}/{acc_base[10:13]}"
    else:
        base_url = f"https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/{acc_base[4:7]}/{acc_base[7:10]}/{acc_base[10:13]}"
    
    # Most recent version typically has format: ACC_ASM###v#
    # We'll try the direct genomic.fna.gz file
    url = f"{base_url}/{acc}_{acc}_genomic.fna.gz"
    
    urls.append({'Assembly': acc, 'URL': url})

df = pd.DataFrame(urls)
df.to_csv("data/ncbi_missing_urls/download_urls.csv", index=False)

print(f"Generated {len(urls):,} download URLs")
print("Saved to data/ncbi_missing_urls/download_urls.csv")

EOFPY

echo ""
echo "Downloading genomes..."
echo "This will take 1-2 hours for 1,277 genomes"
echo ""

# Download using wget (more reliable than Python requests for large files)
cd data/ncbi_missing_fna

# Read URLs and download
tail -n +2 ../ncbi_missing_urls/download_urls.csv | while IFS=',' read assembly url; do
    filename="${assembly}.fna.gz"
    
    if [ ! -f "$filename" ]; then
        echo "Downloading $assembly..."
        curl -s -o "$filename" "$url" 2>/dev/null || wget -q -O "$filename" "$url" 2>/dev/null || echo "  Failed: $assembly"
    fi
done

cd ../..

echo ""
echo "Download complete! Checking results..."

ls data/ncbi_missing_fna/*.fna.gz 2>/dev/null | wc -l

