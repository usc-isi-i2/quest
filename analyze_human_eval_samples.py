import json 
from utils_file import read_jsonl
from pprint import pprint
import numpy as np
import pandas as pd 
from scipy.stats import ttest_ind, pearsonr


def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def format_type_name(qtype):
    """Convert abbreviated type names to full names."""
    return {"util": "Utility", "sal": "Saliency", "eig": "EIG"}.get(qtype, qtype)


def print_means(df, column, title, higher_better=True):
    """Print mean values by question type."""
    print(f"\n{title}:")
    print("-" * 40)
    means = df.groupby("type")[column].mean()
    standard_error = df.groupby("type")[column].std() / np.sqrt(len(df))
    for qtype, value in means.items():
        type_name = format_type_name(qtype)
        print(f"{type_name:>10}: {value:.3f} ± {standard_error[qtype]:.3f}")


def print_ttest_results(df, column, title):
    """Print t-test results for pairwise comparisons."""
    print(f"\n{title}:")
    print("-" * 50)
    
    types = ["util", "sal", "eig"]
    type_names = [format_type_name(t) for t in types]
    
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            type1, type2 = types[i], types[j]
            name1, name2 = type_names[i], type_names[j]
            
            t_stat, p_value = ttest_ind(
                df[df.type == type1][column], 
                df[df.type == type2][column]
            )
            significance = get_significance_stars(p_value)
            print(f"{name1} vs {name2:>10}: t = {t_stat:.3f}, p = {p_value:.4f} {significance}")


def print_correlation_results(df, var1, var2, title=""):
    """Print correlation results between two variables."""
    if title:
        print(f"\n{title}:")
        print("-" * 50)
    
    corr, p_value = pearsonr(df[var1], df[var2])
    significance = get_significance_stars(p_value)
    print(f"{var1.title()} vs {var2.title()}: r = {corr:.4f}, p = {p_value:.4f} {significance}")


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)

annotations = read_jsonl("annotations.jsonl")

qual_feedbacks = [] 
# create a dataframe from annotations 
df_samples =[] 
for ann in annotations: 
    util_q = [q for q in ann["generated_questions"] if q["label"] == "utility"][0]
    sal_q = [q for q in ann["generated_questions"] if q["label"] == "saliency"][0]
    eig_q = [q for q in ann["generated_questions"] if q["label"] == "eig"][0]
    
    util_q_results =[q for q in ann["survey"]["per_question"] if q["label"] == "utility"][0]
    sal_q_results =[q for q in ann["survey"]["per_question"] if q["label"] == "saliency"][0]
    eig_q_results =[q for q in ann["survey"]["per_question"] if q["label"] == "eig"][0]
    
    util_q_sample = {
        "subject": ann["subject"],
        "chapter": ann["chapter"],
        "section_id": ann["section_id"],
        "annotator": ann["annotator"],
        "type": "util",
        "question": util_q["question"],
        "answer": util_q["answer"],
        "utility": util_q["utility"],
        "saliency": util_q["saliency"],
        "eig": util_q["eig"],
        "usefulness": util_q_results["usefulness"],
        "interestingness": util_q_results["interestingness"],
        "rank": util_q_results["rank"],
        "explanation": util_q_results["explanation"],
    }

    sal_q_sample = {
        "subject": ann["subject"],
        "chapter": ann["chapter"],
        "section_id": ann["section_id"],
        "annotator": ann["annotator"],
        "type": "sal",
        "question": sal_q["question"],
        "answer": sal_q["answer"],
        "utility": sal_q["utility"],
        "saliency": sal_q["saliency"],
        "eig": sal_q["eig"],
        "usefulness": sal_q_results["usefulness"],
        "interestingness": sal_q_results["interestingness"],
        "rank": sal_q_results["rank"],
        "explanation": sal_q_results["explanation"],
    }

    eig_q_sample = {
        "subject": ann["subject"],
        "chapter": ann["chapter"],
        "section_id": ann["section_id"],
        "annotator": ann["annotator"],
        "type": "eig",
        "question": eig_q["question"],
        "answer": eig_q["answer"],
        "utility": eig_q["utility"],
        "saliency": eig_q["saliency"],
        "eig": eig_q["eig"],
        "usefulness": eig_q_results["usefulness"],
        "interestingness": eig_q_results["interestingness"],
        "rank": eig_q_results["rank"],
        "explanation": eig_q_results["explanation"],
    }
    
    qual_feedbacks.append({
        "eig_question": eig_q["question"],
        "eig_answer": eig_q["answer"],
        "eig_explanation": eig_q_results["explanation"],
        "sal_question": sal_q["question"],
        "sal_answer": sal_q["answer"],
        "sal_explanation": sal_q_results["explanation"],
        "util_question": util_q["question"],    
        "util_answer": util_q["answer"],
        "util_explanation": util_q_results["explanation"],
    })

    df_samples.append(util_q_sample)
    df_samples.append(sal_q_sample)
    df_samples.append(eig_q_sample)

df = pd.DataFrame(df_samples)


# DESCRIPTIVE STATISTICS
print_section_header("DESCRIPTIVE STATISTICS BY QUESTION TYPE")

print_means(df, "rank", "MEAN RANKING (Lower = Better)")
print_ttest_results(df, "rank", "T-TEST RESULTS (Ranking Differences)")

print_means(df, "usefulness", "MEAN USEFULNESS (Higher = Better)")
print_ttest_results(df, "usefulness", "T-TEST RESULTS (Usefulness Differences)")

print_means(df, "interestingness", "MEAN INTERESTINGNESS (Higher = Better)")
print_ttest_results(df, "interestingness", "T-TEST RESULTS (Interestingness Differences)")
print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

# CORRELATION ANALYSIS
print_section_header("CORRELATION ANALYSIS")

# Primary correlations with human ratings
print("\nCORRELATIONS WITH HUMAN USEFULNESS RATINGS:")
print("-" * 50)
print_correlation_results(df, "utility", "usefulness", "")
print_correlation_results(df, "saliency", "usefulness", "")
print_correlation_results(df, "eig", "usefulness", "")

# Cross-correlations between metrics
print("\nCROSS-CORRELATIONS BETWEEN METRICS:")
print("-" * 50)
print_correlation_results(df, "utility", "saliency", "")
print_correlation_results(df, "utility", "eig", "")
print_correlation_results(df, "saliency", "eig", "")

print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")
print("="*60)


breakpoint() 