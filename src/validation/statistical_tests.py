"""
Add statistical significance testing to all comparisons
Bootstrap confidence intervals, paired t-tests, Bonferroni correction
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm

def bootstrap_auroc(y_true, y_pred_proba, n_bootstrap=1000):
    """
    Compute bootstrap confidence interval for AUROC
    """
    n = len(y_true)
    aurocs = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        aurocs.append(roc_auc_score(y_true[indices], y_pred_proba[indices]))
    
    return {
        'mean': np.mean(aurocs),
        'std': np.std(aurocs),
        'ci_lower': np.percentile(aurocs, 2.5),
        'ci_upper': np.percentile(aurocs, 97.5)
    }

def paired_ttest_cv(X1, X2, y, n_splits=5):
    """
    Paired t-test comparing two models using CV
    Returns p-value for "model 1 != model 2"
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    aurocs_1 = []
    aurocs_2 = []
    
    for train_idx, test_idx in cv.split(X1, y):
        # Model 1
        clf1 = LogisticRegression(max_iter=2000, random_state=42)
        clf1.fit(X1[train_idx], y[train_idx])
        aurocs_1.append(roc_auc_score(y[test_idx], clf1.predict_proba(X1[test_idx])[:, 1]))
        
        # Model 2
        clf2 = LogisticRegression(max_iter=2000, random_state=42)
        clf2.fit(X2[train_idx], y[train_idx])
        aurocs_2.append(roc_auc_score(y[test_idx], clf2.predict_proba(X2[test_idx])[:, 1]))
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(aurocs_1, aurocs_2)
    
    return {
        'mean_1': np.mean(aurocs_1),
        'mean_2': np.mean(aurocs_2),
        'std_1': np.std(aurocs_1),
        'std_2': np.std(aurocs_2),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def compare_pipelines_with_stats():
    """
    Full statistical comparison of all pipelines
    """
    print("="*80)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*80)
    
    # Load embeddings
    Z_standard = np.load("data/ml/Z_all_complete.npy")
    Z_masked = np.load("data/ml/Z_masked_ae.npy")
    
    # Load phenotypes
    pheno = pd.read_csv("data/merged/phenotypes_final.csv")
    
    with open("data/ml/genome_ids_all_complete.txt") as f:
        genome_ids = [line.strip().split('.')[0] for line in f]
    
    genome_to_idx = {gid: i for i, gid in enumerate(genome_ids)}
    
    # Test top antibiotics
    top_abs = pheno.groupby('antibiotic').size().sort_values(ascending=False).head(10).index
    
    results = []
    
    for antibiotic in tqdm(top_abs, desc="Testing antibiotics"):
        ab_data = pheno[pheno['antibiotic'] == antibiotic]
        
        indices = []
        labels = []
        for _, row in ab_data.iterrows():
            gid = row['genome_id']
            if gid in genome_to_idx:
                indices.append(genome_to_idx[gid])
                labels.append(1 if row['phenotype'] == 'R' else 0)
        
        if len(set(labels)) < 2 or len(indices) < 50:
            continue
        
        X_std = Z_standard[indices]
        X_mask = Z_masked[indices]
        y = np.array(labels)
        
        # Paired t-test
        test_result = paired_ttest_cv(X_std, X_mask, y)
        
        results.append({
            'antibiotic': antibiotic,
            'n_samples': len(indices),
            'pipeline1_mean': test_result['mean_1'],
            'pipeline1_std': test_result['std_1'],
            'pipelineA_mean': test_result['mean_2'],
            'pipelineA_std': test_result['std_2'],
            'improvement': test_result['mean_2'] - test_result['mean_1'],
            'p_value': test_result['p_value'],
            'significant': test_result['significant']
        })
    
    df = pd.DataFrame(results)
    
    # Bonferroni correction
    alpha = 0.05
    n_tests = len(df)
    bonferroni_threshold = alpha / n_tests
    
    df['significant_bonferroni'] = df['p_value'] < bonferroni_threshold
    
    print("\n" + "="*80)
    print("RESULTS WITH STATISTICAL SIGNIFICANCE")
    print("="*80)
    
    print(f"\n{'Antibiotic':<30} {'Pipeline1':<15} {'PipelineA':<15} {'Δ':<10} {'p-value':<10} {'Sig?'}")
    print("-"*100)
    
    for _, row in df.iterrows():
        sig_marker = "***" if row['significant_bonferroni'] else ("*" if row['significant'] else "")
        print(f"{row['antibiotic']:<30} "
              f"{row['pipeline1_mean']:.3f}±{row['pipeline1_std']:.3f}  "
              f"{row['pipelineA_mean']:.3f}±{row['pipelineA_std']:.3f}  "
              f"{row['improvement']:+.3f}    "
              f"{row['p_value']:.4f}    "
              f"{sig_marker}")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    overall_improvement = df['improvement'].mean()
    overall_p = stats.ttest_1samp(df['improvement'], 0).pvalue
    
    print(f"Mean improvement: {overall_improvement:+.3f}")
    print(f"P-value (one-sample t-test): {overall_p:.4f}")
    print(f"Significant antibiotics (p<0.05): {df['significant'].sum()}/{len(df)}")
    print(f"Significant after Bonferroni: {df['significant_bonferroni'].sum()}/{len(df)}")
    print(f"Bonferroni threshold: {bonferroni_threshold:.4f}")
    
    # Save
    df.to_csv("results/statistical_comparison.csv", index=False)
    print("\n✓ Saved: results/statistical_comparison.csv")
    
    return df

if __name__ == '__main__':
    compare_pipelines_with_stats()

