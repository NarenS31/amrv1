import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

# Left: DNA sequence with k-mers highlighted
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')

sequence = "ATCGATCGATCG"
ax1.text(5, 2.5, 'DNA Sequence:', ha='center', fontsize=14, weight='bold')
ax1.text(5, 2, sequence, ha='center', fontsize=18, family='monospace', 
         bbox=dict(boxstyle='round', facecolor='#ecf0f1'))

# Highlight k-mers
kmers = ['ATCGAT', 'TCGATC', 'CGATCG']
colors = ['#e74c3c', '#3498db', '#27ae60']
y_pos = 1.2

ax1.text(5, 1.5, 'k=6 extraction:', ha='center', fontsize=12, style='italic')
for i, (kmer, color) in enumerate(zip(kmers, colors)):
    ax1.text(3 + i*2, y_pos, kmer, ha='center', fontsize=14, 
             family='monospace', color='white',
             bbox=dict(boxstyle='round', facecolor=color))

# Right: Feature vector heatmap
ax2.set_title('K-mer Count Vector (4096 dimensions)', fontsize=14, weight='bold')
data = np.random.rand(8, 64) * 100
im = ax2.imshow(data, cmap='YlOrRd', aspect='auto')
ax2.set_xlabel('K-mer Index', fontsize=12)
ax2.set_ylabel('Genome Samples', fontsize=12)
plt.colorbar(im, ax=ax2, label='Count')

plt.tight_layout()
plt.savefig('figures/kmer_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Created figures/kmer_visualization.png")
