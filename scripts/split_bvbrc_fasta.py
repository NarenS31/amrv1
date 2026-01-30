from Bio import SeqIO
import os

os.makedirs("data/bvbrc_genomes", exist_ok=True)

count = 0
with open("data/bvbrc_raw/all_bvbrc_genomes.fasta") as f:
    for record in SeqIO.parse(f, "fasta"):
        genome_id = record.id.split('|')[0].replace('/', '_')
        outfile = f"data/bvbrc_genomes/{genome_id}.fna"
        SeqIO.write([record], outfile, "fasta")
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} genomes...")

print(f"Done! Split {count} genomes")
