import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

roadmap = [
    ("External\nValidation", 2, '#3498db'),
    ("AMRFinder\nBenchmark", 4.5, '#e74c3c'),
    ("Multi-drug\nModeling", 7, '#27ae60'),
    ("Other\nPathogens", 9.5, '#9b59b6'),
]

for label, x, color in roadmap:
    circle = mpatches.Circle((x, 3), 0.6, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, 3, label, ha='center', va='center', fontsize=11, weight='bold', color='white')
    
    if x != roadmap[-1][1]:
        ax.annotate('', xy=(x+1.3, 3), xytext=(x+0.7, 3),
                   arrowprops=dict(arrowstyle='->', lw=3, color='black'))

ax.text(6, 5, 'Research Roadmap', ha='center', fontsize=20, weight='bold')
ax.text(6, 0.5, 'Timeline: 6-12 months', ha='center', fontsize=14, style='italic')

plt.tight_layout()
plt.savefig('figures/research_roadmap.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created figures/research_roadmap.png")
