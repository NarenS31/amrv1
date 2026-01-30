"""
Create clean summary tables for presentation
"""
import pandas as pd
import numpy as np

print("="*80)
print("CREATING SUMMARY TABLES")
print("="*80)

# Table 1: Overall Pipeline Performance
pipeline_summary = pd.DataFrame({
    'Pipeline': [
        'AMRFinder (Rule-based)',
        'Pipeline 1 (Standard AE)',
        'Pipeline A (Masked AE)', 
        'Pipeline B (Diffusion)'
    ],
    'AUROC': ['N/A', '0.703', '0.757', '0.629'],
    'Recall': ['0.754', '~0.68', '~0.65', '~0.63'],
    'Uncertainty': ['No', 'No', 'No', 'Yes'],
    'Robustness_50%': ['Unknown', '0.574', '0.621', '0.621'],
    'Key_Feature': [
        'Hand-crafted rules',
        'Baseline pre-training',
        'Denoising pre-training',
        'Probabilistic predictions'
    ]
})

print("\nTable 1: Pipeline Performance Summary")
print(pipeline_summary.to_string(index=False))
pipeline_summary.to_csv('results/tables/pipeline_summary.csv', index=False)

# Table 2: Per-Antibiotic Performance
comparison = pd.read_csv('results/final_comparison.csv')
per_drug = comparison[['antibiotic', 'n_samples', 'amrfinder_recall', 
                       'pipeline1_auroc', 'pipelineA_auroc']].copy()
per_drug.columns = ['Antibiotic', 'N', 'AMRFinder_Recall', 
                    'Pipeline1_AUROC', 'PipelineA_AUROC']
per_drug = per_drug.sort_values('PipelineA_AUROC', ascending=False)

print("\nTable 2: Per-Antibiotic Performance")
print(per_drug.to_string(index=False))
per_drug.to_csv('results/tables/per_antibiotic_performance.csv', index=False)

# Table 3: Robustness Results
robustness = pd.DataFrame({
    'Corruption_Level': ['0%', '10%', '20%', '30%', '40%', '50%'],
    'Standard_AE': [0.648, 0.635, 0.618, 0.599, 0.586, 0.574],
    'Masked_AE': [0.673, 0.662, 0.652, 0.642, 0.631, 0.621],
    'Degradation_Standard': ['-', '-2.0%', '-4.6%', '-7.6%', '-9.6%', '-11.4%'],
    'Degradation_Masked': ['-', '-1.6%', '-3.1%', '-4.6%', '-6.2%', '-7.7%']
})

print("\nTable 3: Robustness to Data Corruption")
print(robustness.to_string(index=False))
robustness.to_csv('results/tables/robustness_results.csv', index=False)

# Table 4: Dataset Summary
dataset_summary = pd.DataFrame({
    'Dataset': [
        'Total Genomes',
        'Genomes with Phenotypes',
        'Unlabeled Genomes',
        'Total Phenotype Observations',
        'Unique Antibiotics',
        'AMRFinder Outputs'
    ],
    'Count': [
        '21,621',
        '9,110',
        '12,511',
        '15,768',
        '50+',
        '200'
    ]
})

print("\nTable 4: Dataset Summary")
print(dataset_summary.to_string(index=False))
dataset_summary.to_csv('results/tables/dataset_summary.csv', index=False)

print("\n✓ All tables saved to results/tables/")
print("="*80)

