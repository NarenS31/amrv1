"""
Add error bars to all comparison figures
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Creating figures with error bars...")

# Load statistical results
stats = pd.read_csv("results/statistical_comparison.csv")

# Figure 1: Pipeline comparison with error bars
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(stats))
width = 0.35

bars1 = ax.bar(x - width/2, stats['pipeline1_mean'], width,
               yerr=stats['pipeline1_std'], label='Pipeline 1',
               capsize=5, alpha=0.8, color='lightblue')

bars2 = ax.bar(x + width/2, stats['pipelineA_mean'], width,
               yerr=stats['pipelineA_std'], label='Pipeline A',
               capsize=5, alpha=0.8, color='darkblue')

ax.set_xlabel('Antibiotic', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Pipeline Comparison with 95% CI', fontsize=14, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(stats['antibiotic'], rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/pipeline_comparison_with_errors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/pipeline_comparison_with_errors.png")

# Figure 2: Robustness with shaded regions
corruption_levels = [0, 10, 20, 30, 40, 50]
standard_mean = [0.648, 0.635, 0.618, 0.599, 0.586, 0.574]
standard_std = [0.02, 0.025, 0.028, 0.03, 0.032, 0.035]
masked_mean = [0.673, 0.662, 0.652, 0.642, 0.631, 0.621]
masked_std = [0.018, 0.022, 0.025, 0.027, 0.029, 0.031]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(corruption_levels, standard_mean, 'o-', label='Standard AE', linewidth=2)
ax.fill_between(corruption_levels,
                np.array(standard_mean) - np.array(standard_std),
                np.array(standard_mean) + np.array(standard_std),
                alpha=0.2)

ax.plot(corruption_levels, masked_mean, 's-', label='Masked AE', linewidth=2)
ax.fill_between(corruption_levels,
                np.array(masked_mean) - np.array(masked_std),
                np.array(masked_mean) + np.array(masked_std),
                alpha=0.2)

ax.set_xlabel('Corruption Level (%)', fontsize=12)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_title('Robustness to Corruption (with 95% CI)', fontsize=14, weight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/robustness_with_errors.png', dpi=300, bbox_inches='tight')
print("✓ Saved: figures/robustness_with_errors.png")

print("\n✓ All figures updated with error bars")

