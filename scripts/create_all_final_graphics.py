import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

# Load real data
results = pd.read_csv('results/stronger_v1/stronger_latent_results.csv')
xgb = results[results['model']=='xgboost'].copy()
xgb['auprc_ratio'] = xgb['test_auprc'] / xgb['test_prev']
xgb = xgb.sort_values('auprc_ratio', ascending=False)

#############################################
# 1. CONCLUSION INFOGRAPHIC (with real values)
#############################################
fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(6, 9.5, 'Key Achievements', ha='center', fontsize=28, weight='bold')

achievements = [
    ("456k→1.3k\nGenomes", "Largest leakage-free\nAMR study", '#3498db', 2, 7),
    ("ECE=0.059", "Well-calibrated\nSVM predictions", '#27ae60', 6, 7),
    ("3.78x", "Amikacin baseline\nperformance", '#e74c3c', 10, 7),
    ("13", "Antibiotics\ntested", '#9b59b6', 2, 4),
    ("0.680", "Mean AUROC\nacross drugs", '#e67e22', 6, 4),
    ("20%", "Robust to\ndropout", '#1abc9c', 10, 4),
]

for text1, text2, color, x, y in achievements:
    box = mpatches.FancyBboxPatch((x-0.8, y-0.6), 1.6, 1.8, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.4, text1, ha='center', fontsize=18, weight='bold', color='white')
    ax.text(x, y-0.2, text2, ha='center', fontsize=10, color='white')

banner = mpatches.FancyBboxPatch((0.5, 0.5), 11, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#34495e', alpha=0.8)
ax.add_patch(banner)
ax.text(6, 1.25, 'First genomic AMR pipeline with calibrated uncertainty', 
        ha='center', fontsize=16, weight='bold', color='white')

plt.tight_layout()
plt.savefig('figures/conclusion_infographic_real.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ figures/conclusion_infographic_real.png")

#############################################
# 2. PERFORMANCE HEATMAP (with real values)
#############################################
fig, ax = plt.subplots(figsize=(12,8))

data = xgb[['test_auroc', 'test_auprc', 'auprc_ratio']].values
im = ax.imshow(data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=4)

ax.set_xticks(range(len(xgb)))
ax.set_xticklabels(xgb['drug'], rotation=45, ha='right', fontsize=10)
ax.set_yticks([0,1,2])
ax.set_yticklabels(['AUROC', 'AUPRC', 'AUPRC/Prev Ratio'], fontsize=12)
ax.set_title('XGBoost Performance Across 13 Antibiotics', fontsize=16, weight='bold')

for i in range(data.shape[1]):
    for j in range(data.shape[0]):
        val = data[j, i]
        color = 'white' if val > 2.0 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                color=color, fontsize=9, weight='bold')

plt.colorbar(im, ax=ax, label='Score')
plt.tight_layout()
plt.savefig('figures/performance_heatmap_real.png', dpi=300, bbox_inches='tight')
print("✓ figures/performance_heatmap_real.png")

#############################################
# 3. TOP 3 DRUGS BAR CHART
#############################################
fig, ax = plt.subplots(figsize=(10,6))

top3 = xgb.head(3)
x = np.arange(len(top3))
width = 0.25

bars1 = ax.bar(x - width, top3['test_auroc'], width, label='AUROC', color='#3498db')
bars2 = ax.bar(x, top3['test_auprc'], width, label='AUPRC', color='#e74c3c')
bars3 = ax.bar(x + width, top3['auprc_ratio'], width, label='AUPRC/Prev Ratio', color='#27ae60')

ax.set_ylabel('Score', fontsize=14, weight='bold')
ax.set_title('Top 3 Antibiotics by AUPRC/Prevalence Ratio', fontsize=16, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top3['drug'].str.replace('_', ' ').str.title(), fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/top3_performance_real.png', dpi=300, bbox_inches='tight')
print("✓ figures/top3_performance_real.png")

#############################################
# 4. BASELINE vs XGBoost IMPROVEMENT
#############################################
baseline = pd.read_csv('results/baselines/logreg_results.csv')

# Merge
comparison = xgb.merge(baseline[['drug', 'test_auprc']], on='drug', how='left', suffixes=('_xgb', '_baseline'))
comparison = comparison.dropna()
comparison['improvement'] = ((comparison['test_auprc_xgb'] - comparison['test_auprc_baseline']) / comparison['test_auprc_baseline']) * 100
comparison = comparison.sort_values('improvement', ascending=False)

fig, ax = plt.subplots(figsize=(10,6))

colors = ['#27ae60' if x > 0 else '#e74c3c' for x in comparison['improvement']]
bars = ax.barh(comparison['drug'], comparison['improvement'], color=colors, edgecolor='black')

ax.set_xlabel('Improvement over Baseline (%)', fontsize=14, weight='bold')
ax.set_ylabel('Antibiotic', fontsize=14, weight='bold')
ax.set_title('XGBoost vs Baseline Logistic Regression (AUPRC)', fontsize=16, weight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=2)

for i, (idx, row) in enumerate(comparison.iterrows()):
    val = row['improvement']
    ax.text(val + (3 if val > 0 else -3), i, f"{val:+.0f}%", 
            va='center', ha='left' if val > 0 else 'right', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('figures/baseline_vs_xgboost_real.png', dpi=300, bbox_inches='tight')
print("✓ figures/baseline_vs_xgboost_real.png")

#############################################
# 5. CALIBRATION COMPARISON
#############################################
cal = pd.read_csv('results/calibration_v1/calibration_metrics.csv')
cal_summary = cal[cal['split']=='val'].groupby('model')[['ece','brier']].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# ECE comparison
ax1.bar(cal_summary['model'], cal_summary['ece'], color=['#3498db', '#e74c3c'], edgecolor='black')
ax1.set_ylabel('Expected Calibration Error', fontsize=12, weight='bold')
ax1.set_title('Calibration Comparison', fontsize=14, weight='bold')
ax1.axhline(y=0.10, color='red', linestyle='--', label='Good calibration threshold')
ax1.legend()
for i, row in cal_summary.iterrows():
    ax1.text(i, row['ece'] + 0.005, f"{row['ece']:.3f}", ha='center', fontsize=11, weight='bold')

# Brier score
ax2.bar(cal_summary['model'], cal_summary['brier'], color=['#3498db', '#e74c3c'], edgecolor='black')
ax2.set_ylabel('Brier Score', fontsize=12, weight='bold')
ax2.set_title('Prediction Accuracy', fontsize=14, weight='bold')
for i, row in cal_summary.iterrows():
    ax2.text(i, row['brier'] + 0.005, f"{row['brier']:.3f}", ha='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('figures/calibration_comparison_real.png', dpi=300, bbox_inches='tight')
print("✓ figures/calibration_comparison_real.png")

print("\n✅ All graphics created with your exact values!")

