# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import pickle

# # Page config
# st.set_page_config(
#     page_title="AMR Prediction - K. pneumoniae",
#     page_icon="🧬",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
# <style>
# .big-font {
#     font-size:24px !important;
#     font-weight: bold;
# }
# .metric-box {
#     background-color: #f0f2f6;
#     padding: 20px;
#     border-radius: 10px;
#     margin: 10px 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # Sidebar
# st.sidebar.title("🧬 Navigation")
# page = st.sidebar.radio("Go to", [
#     "🏠 Overview",
#     "📊 Data & Methods",
#     "📈 Results",
#     "🔬 Interactive Demo",
#     "📋 Summary"
# ])

# # Load data


# @st.cache_data
# def load_data():
#     results = pd.read_csv('results/stronger_v1/stronger_latent_results.csv')
#     calibration = pd.read_csv('results/calibration_v1/calibration_metrics.csv')
#     diffusion = pd.read_csv(
#         'results/diffusion_latents_v1/diffusion_aug_results.csv')
#     return results, calibration, diffusion


# results_df, cal_df, diff_df = load_data()

# #############################################
# # PAGE 1: OVERVIEW
# #############################################
# if page == "🏠 Overview":
#     st.title("🧬 Genomic Prediction of Antibiotic Resistance")
#     st.markdown(
#         "### *Klebsiella pneumoniae* - Calibrated Machine Learning Pipeline")

#     st.markdown("---")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#         st.metric("Genomes Analyzed", "456,222")
#         st.caption("Largest K. pneumoniae study")
#         st.markdown('</div>', unsafe_allow_html=True)

#     with col2:
#         st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#         st.metric("Unique Strains", "1,325")
#         st.caption("After 99.9% deduplication")
#         st.markdown('</div>', unsafe_allow_html=True)

#     with col3:
#         st.markdown('<div class="metric-box">', unsafe_allow_html=True)
#         st.metric("Calibration Score", "0.086 ECE")
#         st.caption("Well-calibrated uncertainty")
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.markdown("---")

#     st.markdown("## 🎯 Problem")
#     st.markdown("""
#     **Antibiotic-resistant bacteria kill 1.27 million people annually.**

#     *Klebsiella pneumoniae* is a WHO-priority pathogen causing hospital infections.

#     **Current challenge:** Lab testing takes 24-72 hours, delaying treatment.

#     **Our solution:** Predict resistance from genome sequences in hours, not days.
#     """)

#     st.markdown("## 🔬 Key Innovation")
#     st.info("""
#     **First genomic AMR pipeline with calibrated uncertainty:**
#     - Strict deduplication prevents inflated performance
#     - Probabilistic predictions enable selective prediction
#     - Robust to incomplete sequencing data
#     """)

# #############################################
# # PAGE 2: DATA & METHODS
# #############################################
# elif page == "📊 Data & Methods":
#     st.title("📊 Data & Methodology")

#     tab1, tab2, tab3 = st.tabs(
#         ["🗂️ Data Collection", "🧹 Deduplication", "🤖 Model Pipeline"])

#     with tab1:
#         st.markdown("### Data Sources")
#         st.markdown("""
#         - **NCBI Genome Database:** 300,000+ assemblies
#         - **BV-BRC:** 150,000+ assemblies
#         - **Phenotype labels:** 2,330 resistance records across 13 antibiotics
#         """)

#         try:
#             st.image('figures/test_sizes_per_drug.png',
#                      caption='Test samples per antibiotic', use_container_width=True)
#         except:
#             st.warning("Image not found: figures/test_sizes_per_drug.png")

#     with tab2:
#         st.markdown("### Why Deduplication Matters")
#         st.markdown("""
#         **Problem:** Public databases contain massive redundancy from outbreak surveillance.

#         **Solution:** Mash clustering at 99.9% Average Nucleotide Identity (ANI)

#         **Result:** 456,222 genomes → 1,325 unique strains (99.7% reduction)
#         """)

#         try:
#             st.image('figures/dedup_reduction.png',
#                      caption='Genome deduplication', use_container_width=True)
#         except:
#             st.warning("Image not found")

#         st.success("✅ Zero cluster overlap between train/validation/test sets")

#     with tab3:
#         st.markdown("### Feature Engineering Pipeline")

#         col1, col2 = st.columns([1, 1])

#         with col1:
#             st.markdown("""
#             **Step 1:** Extract k-mers (k=31)
#             - Genome → 4^31 possible patterns
#             - Count frequency of each pattern

#             **Step 2:** Random Projection
#             - Compress to 8,192 dimensions
#             - Preserves distances

#             **Step 3:** Autoencoder
#             - Further compress to 256 dimensions
#             - Learns biologically meaningful structure
#             - Distance correlation = 0.83
#             """)

#         with col2:
#             st.markdown("""
#             **Step 4:** Classification
#             - Logistic Regression (baseline)
#             - XGBoost (primary model)
#             - Calibrated SVM (uncertainty)

#             **Step 5:** Uncertainty Quantification
#             - Platt scaling for calibration
#             - Expected Calibration Error < 0.10
#             - Enables selective prediction
#             """)

# #############################################
# # PAGE 3: RESULTS
# #############################################
# elif page == "📈 Results":
#     st.title("📈 Results & Performance")

#     tab1, tab2, tab3, tab4 = st.tabs(
#         ["🎯 Prediction", "📏 Calibration", "🛡️ Robustness", "🔄 Augmentation"])

#     with tab1:
#         st.markdown("### Model Performance by Antibiotic")

#         xgb_results = results_df[results_df['model'] == 'xgboost'].copy()
#         xgb_results['auprc_ratio'] = xgb_results['test_auprc'] / \
#             xgb_results['test_prev']

#         st.dataframe(
#             xgb_results[['drug', 'test_n', 'test_prev',
#                          'test_auroc', 'test_auprc', 'auprc_ratio']]
#             .sort_values('auprc_ratio', ascending=False)
#             .style.format({
#                 'test_n': '{:.0f}',
#                 'test_prev': '{:.3f}',
#                 'test_auroc': '{:.3f}',
#                 'test_auprc': '{:.3f}',
#                 'auprc_ratio': '{:.2f}x'
#             })
#             .background_gradient(subset=['auprc_ratio'], cmap='RdYlGn', vmin=0.5, vmax=5)
#         )

#         st.markdown("""
#         **Interpretation:**
#         - **Amikacin:** 4.98× baseline (real learning)
#         - **High-prevalence drugs:** Near 1.0× (trivial task)
#         """)

#         try:
#             st.image('figures/auprc_vs_prevalence.png',
#                      use_container_width=True)
#         except:
#             st.warning("Image not found")

#     with tab2:
#         st.markdown("### Calibration Analysis")

#         st.markdown("""
#         **Expected Calibration Error (ECE) = 0.086**

#         ✅ ECE < 0.10 indicates well-calibrated predictions

#         When model says "70% confident", it's correct ~70% of the time.
#         """)

#         try:
#             st.image('figures/calibration_all_antibiotics.png',
#                      use_container_width=True)
#         except:
#             st.warning("Image not found")

#     with tab3:
#         st.markdown("### Robustness to Missing Data")

#         st.markdown("""
#         **Simulated incomplete sequencing:** Randomly drop 0-30% of k-mer features

#         **Results:**
#         - Low-prevalence drugs retain signal up to 20% dropout
#         - Performance degrades gracefully beyond 20%
#         - Model learns distributed genomic patterns, not single-gene shortcuts
#         """)

#         try:
#             st.image('figures/robustness_with_errors.png',
#                      use_container_width=True)
#         except:
#             st.warning("Image not found")

#     with tab4:
#         st.markdown("### Diffusion Augmentation Utility")

#         st.markdown("""
#         **Method:** Generate synthetic latent embeddings using diffusion models

#         **When it helps:** Moderately imbalanced drugs (30-70% prevalence)

#         **When it hurts:** Extremely imbalanced drugs
#         """)

#         st.dataframe(
#             diff_df.sort_values('test_auprc_aug', ascending=False)
#             .style.format({'val_auprc_aug': '{:.3f}', 'test_auprc_aug': '{:.3f}'})
#         )

#         try:
#             st.image('results/diffusion_utility_v1/diffusion_gain_vs_pos.png',
#                      use_container_width=True)
#         except:
#             st.warning("Image not found")

# #############################################
# # PAGE 4: INTERACTIVE DEMO
# #############################################
# elif page == "🔬 Interactive Demo":
#     st.title("🔬 Interactive Resistance Prediction")

#     st.markdown("""
#     **Simulate a resistance prediction scenario:**

#     Select an antibiotic and adjust model confidence to see clinical decision outcomes.
#     """)

#     st.markdown("---")

#     # Select antibiotic
#     antibiotics = results_df[results_df['model'] == 'xgboost']['drug'].unique()
#     selected_drug = st.selectbox("Select Antibiotic", antibiotics)

#     # Get performance for selected drug
#     drug_perf = results_df[
#         (results_df['drug'] == selected_drug) &
#         (results_df['model'] == 'xgboost')
#     ].iloc[0]

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("### 📊 Drug Statistics")
#         st.metric("Test AUROC", f"{drug_perf['test_auroc']:.3f}")
#         st.metric("Test AUPRC", f"{drug_perf['test_auprc']:.3f}")
#         st.metric("Test Set Size", f"{int(drug_perf['test_n'])}")
#         st.metric("Prevalence", f"{drug_perf['test_prev']:.1%}")

#     with col2:
#         st.markdown("### 🎚️ Selective Prediction")

#         confidence_threshold = st.slider(
#             "Confidence Threshold",
#             min_value=0.5,
#             max_value=0.95,
#             value=0.7,
#             step=0.05,
#             help="Predictions below this confidence are deferred to lab testing"
#         )

#         # Simulate selective prediction
#         n_test = int(drug_perf['test_n'])
#         base_acc = drug_perf['test_auroc']

#         # Simple model: accuracy improves as you defer uncertain cases
#         frac_deferred = (0.95 - confidence_threshold) / 0.45
#         improved_acc = base_acc + (0.15 * frac_deferred)

#         coverage = 1 - frac_deferred

#         st.metric(
#             "Predicted Accuracy",
#             f"{improved_acc:.1%}",
#             delta=f"+{(improved_acc - base_acc)*100:.1f}%"
#         )
#         st.metric(
#             "Coverage",
#             f"{coverage:.1%}",
#             delta=f"-{frac_deferred*100:.0f}% to lab"
#         )

#         st.info(f"""
#         **Clinical Interpretation:**

#         With threshold {confidence_threshold:.0%}:
#         - {coverage*n_test:.0f}/{n_test} cases use AI prediction
#         - {frac_deferred*n_test:.0f}/{n_test} deferred to lab
#         - Overall accuracy: {improved_acc:.1%}
#         """)

#     st.markdown("---")

#     st.markdown("### 💡 Example Scenario")

#     scenario_col1, scenario_col2 = st.columns(2)

#     with scenario_col1:
#         st.markdown("#### ❌ Without Selective Prediction")
#         st.markdown(f"""
#         - Use model on all {n_test} cases
#         - Accuracy: {base_acc:.1%}
#         - Lab load: 0 samples
#         - **Risk:** {(1-base_acc)*n_test:.0f} misdiagnoses
#         """)

#     with scenario_col2:
#         st.markdown("#### ✅ With Selective Prediction")
#         st.markdown(f"""
#         - Use model on {coverage*n_test:.0f} confident cases
#         - Send {frac_deferred*n_test:.0f} uncertain to lab
#         - Accuracy: {improved_acc:.1%}
#         - **Outcome:** {((improved_acc-base_acc)*n_test):.0f} fewer errors!
#         """)

# #############################################
# # PAGE 5: SUMMARY
# #############################################
# else:
#     st.title("📋 Project Summary")

#     st.markdown("## 🎯 Key Contributions")

#     st.success("""
#     **1. Largest leakage-free AMR genomics study**
#     - 456,222 genomes → 1,325 unique strains
#     - Strict cluster-based splitting (zero overlap)
#     """)

#     st.success("""
#     **2. First calibrated uncertainty quantification**
#     - ECE = 0.086 (well-calibrated)
#     - Enables selective prediction strategies
#     """)

#     st.success("""
#     **3. Honest performance reporting**
#     - Prevalence-normalized metrics
#     - Full distribution across 13 antibiotics
#     - Transparent about limitations
#     """)

#     st.markdown("---")

#     st.markdown("## 📊 Final Results Table")

#     summary = pd.DataFrame({
#         'Metric': [
#             'Genomes Collected',
#             'Unique After Dedup',
#             'Phenotype Records',
#             'Antibiotics Tested',
#             'Mean AUROC',
#             'Calibration (ECE)',
#             'Best Drug (Amikacin)',
#             'Robustness (20% dropout)'
#         ],
#         'Value': [
#             '456,222',
#             '1,325',
#             '2,330',
#             '13',
#             '0.71',
#             '0.086',
#             '4.98× baseline',
#             'Maintained'
#         ]
#     })

#     st.table(summary)

#     st.markdown("---")

#     st.markdown("## 🔬 Methods")
#     with st.expander("Data Processing"):
#         st.markdown("""
#         - K-mer extraction (k=31)
#         - Random projection (8192 dims)
#         - Autoencoder (256 dims)
#         - Mash clustering (99.9% ANI)
#         """)

#     with st.expander("Models"):
#         st.markdown("""
#         - Logistic Regression (baseline)
#         - XGBoost (primary)
#         - Calibrated SVM (uncertainty)
#         - Latent diffusion (augmentation)
#         """)

#     with st.expander("Evaluation"):
#         st.markdown("""
#         - AUPRC (primary metric)
#         - AUROC
#         - Expected Calibration Error
#         - Bootstrap confidence intervals
#         - Robustness testing (k-mer dropout)
#         """)

#     st.markdown("---")

#     st.markdown("## 🚀 Future Work")
#     st.markdown("""
#     - External validation on independent hospital datasets
#     - Comparison to AMRFinder and published baselines
#     - Multi-drug joint modeling
#     - Protein-level feature integration
#     - Expansion to other pathogens (E. coli, P. aeruginosa)
#     """)

#     st.markdown("---")

#     st.markdown("### 👤 Researcher")
#     st.markdown("**Naren Saravanan**")
#     st.markdown("Marvin Ridge High School, North Carolina")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# Page config
st.set_page_config(
    page_title="AMR Prediction - K. pneumoniae",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
    font-weight: bold;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Data & Methods",
    "Results",
    "Interactive Demo",
    "Summary"
])


@st.cache_data
def load_data():
    results = pd.read_csv('results/stronger_v1/stronger_latent_results.csv')
    calibration = pd.read_csv('results/calibration_v1/calibration_metrics.csv')
    diffusion = pd.read_csv(
        'results/diffusion_latents_v1/diffusion_aug_results.csv')
    return results, calibration, diffusion


results_df, cal_df, diff_df = load_data()

#############################################
# PAGE 1: OVERVIEW
#############################################
if page == "Overview":
    st.title("Genomic Prediction of Antibiotic Resistance")
    st.markdown(
        "### *Klebsiella pneumoniae* - Calibrated Machine Learning Pipeline")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Genomes Analyzed", "456,222")
        st.caption("Largest K. pneumoniae study")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Unique Strains", "1,325")
        st.caption("After 99.9% deduplication")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Calibration Score", "0.086 ECE")
        st.caption("Well-calibrated uncertainty")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## Problem")
    st.markdown("""
    **Antibiotic-resistant bacteria kill 1.27 million people annually.**
    
    *Klebsiella pneumoniae* is a WHO-priority pathogen causing hospital infections.
    
    **Current challenge:** Lab testing takes 24-72 hours, delaying treatment.
    
    **Our solution:** Predict resistance from genome sequences in hours, not days.
    """)

    st.markdown("## Key Innovation")
    st.info("""
    **First genomic AMR pipeline with calibrated uncertainty:**
    - Strict deduplication prevents inflated performance
    - Probabilistic predictions enable selective prediction
    - Robust to incomplete sequencing data
    """)

#############################################
# PAGE 2: DATA & METHODS
#############################################
elif page == "Data & Methods":
    st.title("Data & Methodology")

    tab1, tab2, tab3 = st.tabs(
        ["Data Collection", "Deduplication", "Model Pipeline"])

    with tab1:
        st.markdown("### Data Sources")
        st.markdown("""
        - **NCBI Genome Database:** 300,000+ assemblies
        - **BV-BRC:** 150,000+ assemblies
        - **Phenotype labels:** 2,330 resistance records across 13 antibiotics
        """)

        try:
            st.image('figures/test_sizes_per_drug.png',
                     caption='Test samples per antibiotic', use_container_width=True)
        except:
            st.warning("Image not found: figures/test_sizes_per_drug.png")

    with tab2:
        st.markdown("### Why Deduplication Matters")
        st.markdown("""
        **Problem:** Public databases contain massive redundancy from outbreak surveillance.
        
        **Solution:** Mash clustering at 99.9% Average Nucleotide Identity (ANI)
        
        **Result:** 456,222 genomes → 1,325 unique strains (99.7% reduction)
        """)

        try:
            st.image('figures/dedup_reduction.png',
                     caption='Genome deduplication', use_container_width=True)
        except:
            st.warning("Image not found")

        st.success("Zero cluster overlap between train/validation/test sets")

    with tab3:
        st.markdown("### Feature Engineering Pipeline")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            **Step 1:** Extract k-mers (k=31)
            - Genome → 4^31 possible patterns
            - Count frequency of each pattern
            
            **Step 2:** Random Projection
            - Compress to 8,192 dimensions
            - Preserves distances
            
            **Step 3:** Autoencoder
            - Further compress to 256 dimensions
            - Learns biologically meaningful structure
            - Distance correlation = 0.83
            """)

        with col2:
            st.markdown("""
            **Step 4:** Classification
            - Logistic Regression (baseline)
            - XGBoost (primary model)
            - Calibrated SVM (uncertainty)
            
            **Step 5:** Uncertainty Quantification
            - Platt scaling for calibration
            - Expected Calibration Error < 0.10
            - Enables selective prediction
            """)

#############################################
# PAGE 3: RESULTS
#############################################
elif page == "Results":
    st.title("Results & Performance")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Prediction", "Calibration", "Robustness", "Augmentation"])

    with tab1:
        st.markdown("### Model Performance by Antibiotic")

        xgb_results = results_df[results_df['model'] == 'xgboost'].copy()
        xgb_results['auprc_ratio'] = xgb_results['test_auprc'] / \
            xgb_results['test_prev']

        st.dataframe(
            xgb_results[['drug', 'test_n', 'test_prev',
                         'test_auroc', 'test_auprc', 'auprc_ratio']]
            .sort_values('auprc_ratio', ascending=False)
            .style.format({
                'test_n': '{:.0f}',
                'test_prev': '{:.3f}',
                'test_auroc': '{:.3f}',
                'test_auprc': '{:.3f}',
                'auprc_ratio': '{:.2f}x'
            })
            .background_gradient(subset=['auprc_ratio'], cmap='RdYlGn', vmin=0.5, vmax=5)
        )

        st.markdown("""
        **Interpretation:**
        - **Amikacin:** 4.98× baseline (real learning)
        - **High-prevalence drugs:** Near 1.0× (trivial task)
        """)

        try:
            st.image('figures/auprc_vs_prevalence.png',
                     use_container_width=True)
        except:
            st.warning("Image not found")

    with tab2:
        st.markdown("### Calibration Analysis")

        st.markdown("""
        **Expected Calibration Error (ECE) = 0.086**
        
        ECE < 0.10 indicates well-calibrated predictions
        
        When model says "70% confident", it's correct ~70% of the time.
        """)

        try:
            st.image('figures/calibration_all_antibiotics.png',
                     use_container_width=True)
        except:
            st.warning("Image not found")

    with tab3:
        st.markdown("### Robustness to Missing Data")

        st.markdown("""
        **Simulated incomplete sequencing:** Randomly drop 0-30% of k-mer features
        
        **Results:**
        - Low-prevalence drugs retain signal up to 20% dropout
        - Performance degrades gracefully beyond 20%
        - Model learns distributed genomic patterns, not single-gene shortcuts
        """)

        try:
            st.image('figures/robustness_with_errors.png',
                     use_container_width=True)
        except:
            st.warning("Image not found")

    with tab4:
        st.markdown("### Diffusion Augmentation Utility")

        st.markdown("""
        **Method:** Generate synthetic latent embeddings using diffusion models
        
        **When it helps:** Moderately imbalanced drugs (30-70% prevalence)
        
        **When it hurts:** Extremely imbalanced drugs
        """)

        st.dataframe(
            diff_df.sort_values('test_auprc_aug', ascending=False)
            .style.format({'val_auprc_aug': '{:.3f}', 'test_auprc_aug': '{:.3f}'})
        )

        try:
            st.image('results/diffusion_utility_v1/diffusion_gain_vs_pos.png',
                     use_container_width=True)
        except:
            st.warning("Image not found")

#############################################
# PAGE 4: INTERACTIVE DEMO
#############################################
elif page == "Interactive Demo":
    st.title("Interactive Resistance Prediction")

    st.markdown("""
    **Simulate a resistance prediction scenario:**
    
    Select an antibiotic and adjust model confidence to see clinical decision outcomes.
    """)

    st.markdown("---")

    antibiotics = results_df[results_df['model'] == 'xgboost']['drug'].unique()
    selected_drug = st.selectbox("Select Antibiotic", antibiotics)

    drug_perf = results_df[
        (results_df['drug'] == selected_drug) &
        (results_df['model'] == 'xgboost')
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Drug Statistics")
        st.metric("Test AUROC", f"{drug_perf['test_auroc']:.3f}")
        st.metric("Test AUPRC", f"{drug_perf['test_auprc']:.3f}")
        st.metric("Test Set Size", f"{int(drug_perf['test_n'])}")
        st.metric("Prevalence", f"{drug_perf['test_prev']:.1%}")

    with col2:
        st.markdown("### Selective Prediction")

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05
        )

        n_test = int(drug_perf['test_n'])
        base_acc = drug_perf['test_auroc']
        frac_deferred = (0.95 - confidence_threshold) / 0.45
        improved_acc = base_acc + (0.15 * frac_deferred)
        coverage = 1 - frac_deferred

        st.metric("Predicted Accuracy", f"{improved_acc:.1%}")
        st.metric("Coverage", f"{coverage:.1%}")

#############################################
# PAGE 5: SUMMARY
#############################################
else:
    st.title("Project Summary")
    st.markdown("Researcher: Naren Saravanan")
    st.markdown("Marvin Ridge High School, North Carolina")
