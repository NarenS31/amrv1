#!/usr/bin/env python3
"""
Build genome deduplication clusters from Mash results
Uses iterative BFS instead of recursive DFS
"""
import sys
from collections import defaultdict, deque
import gzip

print("Building deduplication clusters...")

# Read duplicate pairs
edges = []
genomes = set()

print("Reading Mash results (26M pairs, will take 2-3 min)...")
with gzip.open('results/mash/near_dupes_99.tsv.gz', 'rt') as f:
    for i, line in enumerate(f):
        if i % 1000000 == 0:
            print(f"  Processed {i:,} pairs...")
        
        parts = line.strip().split('\t')
        g1 = parts[0].split('/')[-1].replace('.fna', '').replace('_genomic', '')
        g2 = parts[1].split('/')[-1].replace('.fna', '').replace('_genomic', '')
        
        # Extract GCF/GCA ID
        g1 = g1.split('_')[0] + '_' + g1.split('_')[1]
        g2 = g2.split('_')[0] + '_' + g2.split('_')[1]
        
        genomes.add(g1)
        genomes.add(g2)
        edges.append((g1, g2))

print(f"\nTotal unique genomes involved: {len(genomes)}")
print(f"Total edges: {len(edges):,}")

# Build adjacency graph
print("\nBuilding adjacency graph...")
graph = defaultdict(set)
for g1, g2 in edges:
    graph[g1].add(g2)
    graph[g2].add(g1)

# Find connected components using BFS (iterative, no recursion)
print("Finding clusters with BFS...")
visited = set()
clusters = []

for start_genome in genomes:
    if start_genome in visited:
        continue
    
    # BFS to find all connected genomes
    cluster = []
    queue = deque([start_genome])
    visited.add(start_genome)
    
    while queue:
        node = queue.popleft()
        cluster.append(node)
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    clusters.append(cluster)
    
    if len(clusters) % 100 == 0:
        print(f"  Found {len(clusters)} clusters so far...")

print(f"\nFound {len(clusters)} total clusters")

# Sort clusters by size
clusters.sort(key=len, reverse=True)

print("\nTop 10 largest clusters:")
for i, cluster in enumerate(clusters[:10], 1):
    print(f"  Cluster {i}: {len(cluster)} genomes")

# Select one representative per cluster (first genome alphabetically)
keep_genomes = [sorted(cluster)[0] for cluster in clusters]

print(f"\nTotal genomes to KEEP: {len(keep_genomes)}")
print(f"Total genomes REMOVED: {len(genomes) - len(keep_genomes)}")
print(f"Reduction: {(1 - len(keep_genomes)/len(genomes))*100:.1f}%")

# Save keep list
with open('results/mash/keep_genomes.txt', 'w') as f:
    for genome in sorted(keep_genomes):
        f.write(genome + '\n')

print("\n✓ Saved: results/mash/keep_genomes.txt")

# Save cluster info
with open('results/mash/cluster_info.txt', 'w') as f:
    f.write(f"Total clusters: {len(clusters)}\n")
    f.write(f"Genomes kept: {len(keep_genomes)}\n")
    f.write(f"Genomes removed: {len(genomes) - len(keep_genomes)}\n")
    f.write(f"\nTop 50 largest clusters:\n")
    for i, cluster in enumerate(clusters[:50], 1):
        f.write(f"Cluster {i}: {len(cluster)} genomes\n")

print("✓ Saved: results/mash/cluster_info.txt")

