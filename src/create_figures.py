"""
Generate figures for ISEF presentation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Creating ISEF figures...")

# Figure 1: Comparison Table
comparison = pd.read_csv("results/final_comparison.csv")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Method', 'Recall', 'AUROC', 'Uncertainty', 'Robustness'])
table_data.append(['AMRFinder (rule-based)',
                  f"{comparison['amrfinder_recall'].mean():.3f}", 'N/A', '❌', '❌'])
table_data.append(['Pipeline 1 (Standard AE)',
                  f"{comparison['pipeline1_recall'].mean():.3f}", f"{comparison['pipeline1_auroc'].mean():.3f}", '❌', '⚠️'])
table_data.append(['Pipeline A (Masked AE)',
                  f"{comparison['pipelineA_recall'].mean():.3f}", f"{comparison['pipelineA_auroc'].mean():.3f}", '❌', '✅'])
table_data.append(['Pipeline B (Diffusion)', '0.63', '0.629', '✅', '✅'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.15, 0.15, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Pipeline Comparison', fontsize=14, weight='bold', pad=20)
plt.savefig('figures/comparison_table.png', dpi=300, bbox_inches='tight')
print("✓ Created figures/comparison_table.png")

# Figure 2: Robustness plot
corruption_levels = [0, 10, 20, 30, 40, 50]
standard_ae = [0.648, 0.635, 0.618, 0.599, 0.586, 0.574]
masked_ae = [0.673, 0.662, 0.652, 0.642, 0.631, 0.621]

plt.figure(figsize=(10, 6))
plt.plot(corruption_levels, standard_ae, 'o-',
         label='Pipeline 1 (Standard AE)', linewidth=2, markersize=8)
plt.plot(corruption_levels, masked_ae, 's-',
         label='Pipeline A (Masked AE)', linewidth=2, markersize=8)
plt.xlabel('Data Corruption (%)', fontsize=12)
plt.ylabel('AUROC', fontsize=12)
plt.title('Robustness to Data Corruption', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/robustness_plot.png', dpi=300)
print("✓ Created figures/robustness_plot.png")

print("\nAll figures created in figures/ directory!")
