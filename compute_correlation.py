# compute spearman correlation and p value between metrics

from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from scipy.stats import spearmanr, norm
from itertools import combinations

def compute_correlation(results_df, metrics):
    for metric_i, metric_j in combinations(metrics, 2):
            correlation, p_value = spearmanr(results_df[metric_i], results_df[metric_j])
            logger.info(f"{metric_i} and {metric_j}: {correlation:.3f}, p-value: {p_value:.3f}")

def fisher_z_transform(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return (np.exp(2*z) - 1) / (np.exp(2*z) + 1)

def meta_analyze_correlations(correlations, sample_sizes):
    # Transform correlations to z-scores
    z_scores = [fisher_z_transform(r) for r in correlations]
    
    # Weight by sample size
    weights = [n - 3 for n in sample_sizes]  # -3 for correlation degrees of freedom
    
    # Weighted average z-score
    weighted_z = np.sum([w * z for w, z in zip(weights, z_scores)]) / np.sum(weights)
    
    # Standard error
    se = 1 / np.sqrt(np.sum(weights))
    
    # Z-test
    z_stat = weighted_z / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # Transform back to correlation
    avg_correlation = inverse_fisher_z(weighted_z)
    
    return avg_correlation, p_value, z_stat

metric_results = Path("q_metrics").glob("*.jsonl")

metrics = ["saliency", "eig", "utility"]

all_results = []
subject_correlations = {"saliency_eig": [], "saliency_utility": [], "eig_utility": []}
subject_sample_sizes = []

for file in metric_results:
    df = pd.read_json(file, lines=True)
    logger.info(f"Processing {file}")
    compute_correlation(df, metrics)
    
    # Store correlations and sample sizes for meta-analysis
    subject_sample_sizes.append(len(df))
    
    # Calculate and store correlations for each metric pair
    for metric_i, metric_j in combinations(metrics, 2):
        correlation, _ = spearmanr(df[metric_i], df[metric_j])
        pair_name = f"{metric_i}_{metric_j}"
        if pair_name in subject_correlations:
            subject_correlations[pair_name].append(correlation)
    
    all_results.append(df)

# Meta-analysis for each metric pair
logger.info("Meta-analysis of within-subject correlations:")
for pair_name, correlations in subject_correlations.items():
    if len(correlations) > 0:
        avg_corr, p_val, z_stat = meta_analyze_correlations(correlations, subject_sample_sizes)
        logger.info(f"{pair_name}: average correlation = {avg_corr:.3f}, p-value = {p_val:.3f}, z-stat = {z_stat:.3f}")
            
# total number of questions that were used in the meta-analysis
logger.info(f"Total number of questions that were used in the meta-analysis: {len(pd.concat(all_results))}")