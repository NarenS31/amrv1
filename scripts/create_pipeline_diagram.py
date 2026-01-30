import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14,6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Pipeline boxes
boxes = [
    (1, 'Genome\nSequence', '#3498db'),
    (3.5, 'K-mer\nExtraction', '#9b59b6'),
    (6, 'Random\nProjection', '#e67e22'),
    (8.5, 'Autoencoder\nEmbedding', '#e74c3c'),
    (11, 'XGBoost\nClassifier', '#27ae60'),
]

for x, label, color in boxes:
    box = FancyBboxPatch((x-0.5, 2), 1.2, 2, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x+0.1, 3, label, ha='center', va='center', 
            fontsize=11, weight='bold', color='white')

# Arrows
for i in range(len(boxes)-1):
    x_start = boxes[i][0] + 0.7
    x_end = boxes[i+1][0] - 0.5
    arrow = FancyArrowPatch((x_start, 3), (x_end, 3),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color='black')
    ax.add_patch(arrow)

# Dimensions below
dims = ['ATCG...', '4^31', '8192', '256', 'R/S']
for i, (x, _, _) in enumerate(boxes):
    ax.text(x+0.1, 1.5, dims[i], ha='center', fontsize=10, 
            style='italic', color='gray')

# Final output arrow
arrow = FancyArrowPatch((12.2, 3), (13.5, 3),
                       arrowstyle='->', mutation_scale=30, 
                       linewidth=3, color='#27ae60')
ax.add_patch(arrow)
ax.text(13.2, 3.8, 'Calibrated\nProbability', ha='center', 
        fontsize=11, weight='bold', color='#27ae60')

plt.tight_layout()
plt.savefig('figures/pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created figures/pipeline_diagram.png")
