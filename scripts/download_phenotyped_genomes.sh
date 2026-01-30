#!/bin/bash

# Download genomes that have phenotype labels
mkdir -p data/phenotyped_genomes

while read genome_id; do
    echo "Downloading $genome_id..."
    
    datasets download genome accession "$genome_id" \
        --filename "data/phenotyped_genomes/${genome_id}.zip" \
        --include genome
    
    # Unzip
    unzip -q "data/phenotyped_genomes/${genome_id}.zip" -d "data/phenotyped_genomes/${genome_id}/"
    rm "data/phenotyped_genomes/${genome_id}.zip"
    
done < genomes_to_download_versioned.txt

