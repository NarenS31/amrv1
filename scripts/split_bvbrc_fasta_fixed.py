from Bio import SeqIO
import os

os.makedirs("data/bvbrc_genomes_split", exist_ok=True)

count = 0
current_genome = []
current_id = None

with open("data/bvbrc_raw/all_bvbrc_genomes.fasta") as f:
    for line in f:
        if line.startswith('>'):
            # Save previous genome
            if current_genome and current_id:
                safe_id = current_id.split()[0].replace('|', '_').replace('/', '_')
                outfile = f"data/bvbrc_genomes_split/{safe_id}.fna"
                with open(outfile, 'w') as out:
                    out.write(''.join(current_genome))
                count += 1
                if count % 10000 == 0:
                    print(f"Processed {count} genomes...")
            
            # Start new genome
            current_id = line[1:].strip()
            current_genome = [line]
        else:
            current_genome.append(line)
    
    # Save last genome
    if current_genome and current_id:
        safe_id = current_id.split()[0].replace('|', '_').replace('/', '_')
        outfile = f"data/bvbrc_genomes_split/{safe_id}.fna"
        with open(outfile, 'w') as out:
            out.write(''.join(current_genome))
        count += 1

print(f"Done! Split {count} genome files")
