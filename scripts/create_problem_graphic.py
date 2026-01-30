import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9, '1.27 Million Deaths Annually', 
        ha='center', fontsize=32, weight='bold', color='#e74c3c')

# Icon grid representing deaths
import numpy as np
x_pos = np.linspace(1, 9, 10)
y_pos = np.linspace(7, 2, 8)

for x in x_pos[:6]:
    for y in y_pos:
        circle = mpatches.Circle((x, y), 0.15, color='#e74c3c', alpha=0.8)
        ax.add_patch(circle)

# Stats boxes
box1 = mpatches.FancyBboxPatch((0.5, 0.5), 4, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#3498db', alpha=0.3)
ax.add_patch(box1)
ax.text(2.5, 1.1, '24-72 hours', ha='center', fontsize=20, weight='bold')
ax.text(2.5, 0.7, 'Current lab testing delay', ha='center', fontsize=12)

box2 = mpatches.FancyBboxPatch((5.5, 0.5), 4, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#27ae60', alpha=0.3)
ax.add_patch(box2)
ax.text(7.5, 1.1, '<6 hours', ha='center', fontsize=20, weight='bold')
ax.text(7.5, 0.7, 'Genomic prediction goal', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('figures/problem_deaths.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created figures/problem_deaths.png")
