import pandas as pd
import numpy as np

# Load all results
results = pd.read_csv('results/stronger_v1/stronger_latent_results.csv')
baseline = pd.read_csv('results/baselines/logreg_results.csv')
calibration = pd.read_csv('results/calibration_v1/calibration_metrics.csv')

print("="*80)
print("EXACT VALUES FROM YOUR FILES")
print("="*80)

# XGBoost results
xgb = results[results['model']=='xgboost'].copy()
xgb['auprc_ratio'] = xgb['test_auprc'] / xgb['test_prev']
xgb = xgb.sort_values('auprc_ratio', ascending=False)

print("\nXGBOOST PERFORMANCE (sorted by AUPRC/Prevalence ratio):")
print("-"*80)
for _, row in xgb.iterrows():
    print(f"{row['drug']:<30} | Test N: {int(row['test_n']):>4} | "
          f"Prev: {row['test_prev']:>5.1%} | AUROC: {row['test_auroc']:.3f} | "
          f"AUPRC: {row['test_auprc']:.3f} | Ratio: {row['auprc_ratio']:.2f}x")

print("\n" + "="*80)
print("CALIBRATION SCORES:")
print("-"*80)
cal_summary = calibration[calibration['split']=='val'].groupby('model')[['ece','brier']].mean()
print(cal_summary)

print("\n" + "="*80)
print("TOP 3 ANTIBIOTICS (by ratio):")
print("-"*80)
top3 = xgb.head(3)
for _, row in top3.iterrows():
    print(f"\n{row['drug'].upper()}:")
    print(f"  - Test samples: {int(row['test_n'])}")
    print(f"  - Prevalence: {row['test_prev']:.1%}")
    print(f"  - AUROC: {row['test_auroc']:.3f}")
    print(f"  - AUPRC: {row['test_auprc']:.3f}")
    print(f"  - AUPRC/Prev ratio: {row['auprc_ratio']:.2f}x baseline")

print("\n" + "="*80)
print("OVERALL STATISTICS:")
print("-"*80)
print(f"Mean AUROC: {xgb['test_auroc'].mean():.3f}")
print(f"Median AUROC: {xgb['test_auroc'].median():.3f}")
print(f"Mean AUPRC/Prev ratio: {xgb['auprc_ratio'].mean():.2f}x")
print(f"Best ratio (amikacin): {xgb['auprc_ratio'].max():.2f}x")

print("\n" + "="*80)
print("BASELINE (Logistic Regression on raw k-mers) vs LATENT MODELS:")
print("-"*80)
# Compare to baseline
for drug in xgb['drug'].head(5):
    xgb_perf = xgb[xgb['drug']==drug]['test_auprc'].values[0]
    if drug in baseline['drug'].values:
        base_perf = baseline[baseline['drug']==drug]['test_auprc'].values[0]
        improvement = ((xgb_perf - base_perf) / base_perf) * 100
        print(f"{drug:<25}: Baseline {base_perf:.3f} → XGBoost {xgb_perf:.3f} "
              f"({improvement:+.1f}% change)")

