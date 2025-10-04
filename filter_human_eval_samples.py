
from pathlib import Path 
import json 
import pandas as pd
import numpy as np
from utils_file import read_jsonl
from loguru import logger

q_metrics_dir = Path("output/gpt-4o-mini")
q_metrics_files = list(q_metrics_dir.glob("*_m*.jsonl"))

qa_pairs = [] 
for file in q_metrics_files:
    with open(file, "r") as f:
        for line in f:
            qa_pairs.append(json.loads(line))

textbook_data = read_jsonl("data/data.jsonl")

df = pd.DataFrame(qa_pairs)
subjects= df.subject.unique()

human_eval_samples = [] 
for subject in subjects:
    subject_df = df[df.subject == subject]

    # for each section, get 1 high utility question, 1 high saliency question, 1 high eig question (but not high for other metrics)

    percentile = 80

    # use top 20% percentile of each metric as threshold for high utility, saliency, eig
    utility_threshold = max(np.percentile(subject_df.utility, percentile), 0.1)
    saliency_threshold = np.percentile(subject_df.saliency, percentile)
    eig_threshold = np.percentile(subject_df.eig, percentile)
 
    logger.info(f"Thresholds for {subject}: utility={utility_threshold:.4f}, saliency={saliency_threshold:.4f}, eig={eig_threshold:.4f}")
 
    for chapter in subject_df.chapter.unique():
        chapter_df = subject_df[subject_df.chapter == chapter]
        
        for section in chapter_df.section.unique():
            section_df = chapter_df[chapter_df.section == section]

            high_utility_questions = section_df[(section_df.utility >= utility_threshold) & (section_df.saliency <= saliency_threshold) & (section_df.eig <= eig_threshold)]
            high_saliency_questions = section_df[(section_df.saliency >= saliency_threshold) & (section_df.utility <= utility_threshold) & (section_df.eig <= eig_threshold)]
            high_eig_questions = section_df[(section_df.eig >= eig_threshold) & (section_df.utility <= utility_threshold) & (section_df.saliency <= saliency_threshold)]
            
            if len(high_utility_questions) == 0: 
                continue
            if len(high_saliency_questions) == 0: 
                continue
            if len(high_eig_questions) == 0:
                continue
                    
                    
            high_utility_questions, high_saliency_questions, high_eig_questions
            
            human_eval_samples.append({
                "subject": subject,
                "chapter": chapter,
                "section": int(section),
                "high_utility_question": {
                    "question": high_utility_questions.iloc[0]["question"],
                    "answer": high_utility_questions.iloc[0]["answer"],
                    "utility": round(float(high_utility_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_utility_questions.iloc[0]["saliency"]), 4),
                    "eig": round(float(high_utility_questions.iloc[0]["eig"]), 4),
                },
                "high_saliency_question": {
                    "question": high_saliency_questions.iloc[0]["question"],
                    "answer": high_saliency_questions.iloc[0]["answer"],
                    "utility": round(float(high_saliency_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_saliency_questions.iloc[0]["saliency"]), 4),
                    "eig": round(float(high_saliency_questions.iloc[0]["eig"]), 4),
                },
                "high_eig_question": {
                    "question": high_eig_questions.iloc[0]["question"],
                    "answer": high_eig_questions.iloc[0]["answer"],
                    "utility": round(float(high_eig_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_eig_questions.iloc[0]["saliency"]), 4),
                    "eig": round(float(high_eig_questions.iloc[0]["eig"]), 4),
                },
            })

# number of unique subject chapter section combinations
unique_subject_chapter_section_combinations = df[['subject', 'chapter', 'section']].drop_duplicates().shape[0]
logger.info(f"Total human eval samples: {len(human_eval_samples)} from {unique_subject_chapter_section_combinations} sections")

with open("data/human_eval_samples.jsonl", "w") as f:
    for sample in human_eval_samples:
        f.write(json.dumps(sample) + "\n")
        
        
# compute correlations between utility, saliency, eig in df 
# metrics = ["saliency", "eig", "utility"]
# from compute_correlation import compute_correlation
# compute_correlation(df, metrics)