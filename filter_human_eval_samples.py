
from pathlib import Path 
import json 
import pandas as pd
import numpy as np
from utils_file import read_jsonl
from loguru import logger

from argparse import ArgumentParser
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--add_qsalience_score", action="store_true", help="Add qsalience score to human eval samples")
    parser.add_argument("--percentile", type=int, default=80, help="Percentile to use for high utility, saliency, eig")
    parser.add_argument("--qsalience_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="model name for qsalience")
    parser.add_argument("--qsalience_qlora_model_name", type=str, default="lingchensanwen/mistral-ins-generation-best-balanced", help="qlora model name for qsalience")
    return parser.parse_args()
args = parse_args()


q_metrics_dir = Path("output/gpt-4o-mini")
q_metrics_files = list(q_metrics_dir.glob("*_m*.jsonl"))

if args.add_qsalience_score:
    logger.info("Loading QSalience model (this may take 30-60 seconds)...")
    from utils_qsalience import QSalience
    qsalience = QSalience(args.qsalience_model_name, args.qsalience_qlora_model_name)
    logger.info("✅ QSalience model loaded successfully")

qa_pairs = [] 
for file in q_metrics_files:
    current_qa_pairs = [] 
    with open(file, "r") as f:
        current_qa_pairs = [json.loads(line) for line in f]
          
    if args.add_qsalience_score:
        for qa in tqdm(current_qa_pairs): 
            if "qsaliency" not in qa or qa["qsaliency"] < 1 or qa["qsaliency"] > 5:
                qa["qsaliency"] = qsalience.predict_salience(qa["context"] + qa["anchor"], qa["question"])
        saliency_metric = "qsaliency"
        
        with open(file, "w") as f: 
            for qa in current_qa_pairs:
                f.write(json.dumps(qa) + "\n")
    else:
        saliency_metric = "saliency"

    qa_pairs.extend(current_qa_pairs)

textbook_data = read_jsonl("data/data.jsonl")

df = pd.DataFrame(qa_pairs)
if saliency_metric == "qsaliency":
    # drop cases where qsalience model failed to assign a proper score, which should be between 1 and 5 
    original_df_length = len(df)
    df = df[df.qsaliency >= 1]
    df = df[df.qsaliency <= 5]
    
    logger.info(f"Dropped {original_df_length - len(df)} cases where qsalience model failed to assign a proper score")

subjects= df.subject.unique()

human_eval_samples = [] 
for subject in subjects:
    subject_df = df[df.subject == subject]

    # for each section, get 1 high utility question, 1 high saliency question, 1 high eig question (but not high for other metrics)

    percentile = args.percentile

    # use top 20% percentile of each metric as threshold for high utility, saliency, eig
    utility_threshold = max(np.percentile(subject_df.utility, percentile), 0.1)
    saliency_threshold = np.percentile(subject_df[saliency_metric], percentile)
    eig_threshold = np.percentile(subject_df.eig, percentile)
 
    logger.info(f"Thresholds for {subject}: utility={utility_threshold:.4f}, saliency={saliency_threshold:.4f}, eig={eig_threshold:.4f}")
 
    for chapter in subject_df.chapter.unique():
        chapter_df = subject_df[subject_df.chapter == chapter]
        
        for section in chapter_df.section.unique():
            section_df = chapter_df[chapter_df.section == section]

            high_utility_questions = section_df[(section_df.utility >= utility_threshold) & (section_df[saliency_metric] <= saliency_threshold) & (section_df.eig <= eig_threshold)]
            high_saliency_questions = section_df[(section_df[saliency_metric] >= saliency_threshold) & (section_df.utility <= utility_threshold) & (section_df.eig <= eig_threshold)]
            # sort by lowest utility 
            high_utility_questions = high_utility_questions.sort_values(by="utility")    
        
            high_eig_questions = section_df[(section_df.eig >= eig_threshold) & (section_df.utility <= utility_threshold) & (section_df.saliency <= saliency_threshold)]
            # sort by lowest utility 
            high_saliency_questions = high_saliency_questions.sort_values(by="utility")
            
            if len(high_utility_questions) == 0: 
                continue
            if len(high_saliency_questions) == 0: 
                continue
            if len(high_eig_questions) == 0:
                continue
            
            # make sure that the questions are unique 
            if high_utility_questions.iloc[0]["question"] == high_saliency_questions.iloc[0]["question"]:
                continue
            if high_utility_questions.iloc[0]["question"] == high_eig_questions.iloc[0]["question"]:
                continue
            if high_saliency_questions.iloc[0]["question"] == high_eig_questions.iloc[0]["question"]:
                continue
            
            human_eval_samples.append({
                "subject": subject,
                "chapter": chapter,
                "section_id": int(section_df.section.iloc[0]),
                "context": section_df.context.iloc[0],
                "anchor": section_df.anchor.iloc[0],
                "relevant_questions": section_df.relevant_questions.iloc[0],
                "section": int(section),
                "high_utility_question": {
                    "question": high_utility_questions.iloc[0]["question"],
                    "answer": high_utility_questions.iloc[0]["answer"],
                    "utility": round(float(high_utility_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_utility_questions.iloc[0][saliency_metric]), 4),
                    "eig": round(float(high_utility_questions.iloc[0]["eig"]), 4),
                },
                "high_saliency_question": {
                    "question": high_saliency_questions.iloc[0]["question"],
                    "answer": high_saliency_questions.iloc[0]["answer"],
                    "utility": round(float(high_saliency_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_saliency_questions.iloc[0][saliency_metric]), 4),
                    "eig": round(float(high_saliency_questions.iloc[0]["eig"]), 4),
                },
                "high_eig_question": {
                    "question": high_eig_questions.iloc[0]["question"],
                    "answer": high_eig_questions.iloc[0]["answer"],
                    "utility": round(float(high_eig_questions.iloc[0]["utility"]), 4),
                    "saliency": round(float(high_eig_questions.iloc[0][saliency_metric]), 4),
                    "eig": round(float(high_eig_questions.iloc[0]["eig"]), 4),
                },
            })

# number of unique subject chapter section combinations
unique_subject_chapter_section_combinations = df[['subject', 'chapter', 'section']].drop_duplicates().shape[0]
logger.info(f"Total human eval samples: {len(human_eval_samples)} from {unique_subject_chapter_section_combinations} sections")

with open(f"data/human_eval_samples_{saliency_metric}.jsonl", "w") as f:
    for sample in human_eval_samples:
        f.write(json.dumps(sample) + "\n")
        
        
# compute correlations between utility, saliency, eig in df 
# metrics = ["saliency", "eig", "utility"]
# from compute_correlation import compute_correlation
# compute_correlation(df, metrics)