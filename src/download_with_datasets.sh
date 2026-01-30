#!/bin/bash

echo "DOWNLOADING GENOMES USING NCBI DATASETS"
echo "========================================"

# Clean up bad downloads
rm -rf data/ncbi_missing_fna/*.fna.gz

# Create accession list
tail -n +2 data/ncbi_missing/biosample_to_assembly.csv | cut -d',' -f2 > data/ncbi_missing/accessions.txt

echo "Accessions to download: $(wc -l < data/ncbi_missing/accessions.txt)"

# Download in batches (datasets has limits)
cd data/ncbi_missing

split -l 100 accessions.txt batch_

batch_count=$(ls batch_* | wc -l)
echo "Split into $batch_count batches"

for batch in batch_*; do
    echo "Downloading $batch..."
    datasets download genome accession --inputfile $batch --filename ${batch}.zip
    
    if [ -f "${batch}.zip" ]; then
        echo "Extracting..."
        unzip -q ${batch}.zip -d ${batch}_data
        
        # Move fna files
        find ${batch}_data -name "*.fna" -exec mv {} ../ncbi_missing_fna/ \;
        
        # Clean up
        rm -rf ${batch}_data ${batch}.zip
    fi
done

cd ../..

# Count results
fna_count=$(ls data/ncbi_missing_fna/*.fna 2>/dev/null | wc -l)
echo ""
echo "Downloaded $fna_count genome files"

