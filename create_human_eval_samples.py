# find question sets, one with high utility, one with high saliency, one with high eig
# get the textbook section for that question set from data/data.jsonl
# get the end of chapter questions that are related to that section 

import json 
from pathlib import Path 
from utils_file import read_jsonl
import pandas as pd

import argparse
import os
import random

from agents import AnswerGenerator, QuestionGenerator, Simulator
from utils_file import read_jsonl, write_jsonl
from utils_llm import OpenAIGenerator
from loguru import logger
from run_eval import get_saliency_score, compute_eig_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate questions using different prompting strategies."
    )

    parser.add_argument("--qg_model_name", type=str, default="gpt-4o-mini", help="Model name for question generation")
    parser.add_argument("--evaluate_model_name", type=str, default="gpt-4o-mini", help="Model name for evaluation")
    parser.add_argument(
        "--mode", type=str, choices=["default", "cot", "fewshot"], default="default", help="Prompting strategy"
    )
    parser.add_argument("--subject", type=str, required=True, help="Single subject to process, one of [chemistry, sociology, us-history, microbiology, economics]")
    parser.add_argument("-ud", "--use_document_for_simulate", action="store_true", help="Use document sections in simulation")
    parser.add_argument("-nq", "--num_questions_per_section", type=int, default=1, help="Questions to generate per section")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and initialize base LLM
    llm_generator = OpenAIGenerator(oai_api_key=os.getenv("OPENAI_API_KEY", ""), model=args.evaluate_model_name)

    # Initialize agents
    answer_generator = AnswerGenerator(generator=llm_generator)
    simulator = Simulator(generator=llm_generator)

    # Select appropriate question generator
    question_llm_generator = OpenAIGenerator(oai_api_key=os.getenv("OPENAI_API_KEY", ""), model=args.qg_model_name)
    question_generator = QuestionGenerator(generator=question_llm_generator)

    # Read data
    data = read_jsonl("data/data.jsonl")
    subject = args.subject

    # Create output paths
    os.makedirs(f"output/{args.qg_model_name}", exist_ok=True)
    qa_output_file = f"output/{args.qg_model_name}/{subject}_{args.mode}_qa_pairs.jsonl"
    question_generator_performances = []

    few_shot_candidates = []

    filtered_data = [item for item in data if item["subject"] == subject]
    if not filtered_data:
        logger.info(f"No data found for subject '{subject}'")
        return

    filtered_data = sorted(filtered_data, key=lambda x: x["chapter"])
    train_filtered_data = filtered_data[:-5]
    test_filtered_data = filtered_data[-5:]

    if args.mode == "fewshot":
        few_shot_raw_examples = []
        for d in test_filtered_data:
            sections = d["llm_parsed_results"]["sections"]
            questions = d["llm_parsed_results"]["questions"]
            for sec_id, sec in sections.items():
                anchor = sec["content"]
                context = "\n".join([s["content"] for i, s in sections.items() if int(i) < int(sec_id)])
                for q in questions.values():
                    few_shot_raw_examples.append(
                        {
                            "context": context,
                            "anchor": anchor,
                            "question": q["question"],
                        }
                    )
        random.shuffle(few_shot_raw_examples)
        few_shot_candidates = few_shot_raw_examples[:5]

    for train_data in train_filtered_data:
        chapter = train_data["chapter"]
        chapter_qa_pairs = []
        llm_parsing = train_data["llm_parsed_results"]
        exam_questions = llm_parsing["questions"]

        context = ""
        questions_by_section = {}

        for i in range(1, len(llm_parsing["sections"]) + 1):
            if i > 3: 
                continue 
                        
            anchor = llm_parsing["sections"][str(i)]["content"]
            
            # make sure there's at least 1 exam question that is related to this section             
            relevant_questions = [q for q in llm_parsing["questions"].values() if str(i) in q["relevant_sections"]]
            if len(relevant_questions) == 0:
                logger.info(f"No relevant questions for section {i}, skipping")            
                generated_qa_pairs = []
            else: 
                generated_questions, _ = question_generator.generate(
                    anchor=anchor,
                    context=context,
                    num_questions=args.num_questions_per_section,
                    mode=args.mode,
                    few_shot_candidates=few_shot_candidates if args.mode == "fewshot" else None,
                )
                # logger.info(f"Generated questions: {generated_questions}")
                
                generated_qa_pairs = answer_generator.generate(
                    anchor=anchor, context=context, questions=generated_questions
                )
                questions_by_section[str(i)] = generated_qa_pairs

            context += f"\n{anchor}"

            for idx, qa in enumerate(generated_qa_pairs):
                chapter_qa_pairs.append(
                    {
                        "subject": subject,
                        "chapter": chapter,
                        "anchor": anchor,
                        "context": context,
                        "section": i,
                        "qid": f"{i}_Q{idx+1}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "relevant_questions": relevant_questions,
                    }
                )
        
        if questions_by_section:
        
            if args.use_document_for_simulate:
                _, all_score, baseline_score = simulator.generate(
                    eval_questions=exam_questions,
                    sections=llm_parsing["sections"],
                    generated_questions=questions_by_section,
                    test=True,
                    single_only=True,
                )
            else:
                utilities, _, baseline_score = simulator.generate(
                    eval_questions=exam_questions,
                    sections={},
                    generated_questions=questions_by_section,
                    test=True,
                    single_only=True,
                )
                
        for qa in chapter_qa_pairs:
            qa["utility"] = utilities[qa["qid"]]["utility"]
            if "saliency" not in qa:    
                qa["saliency"] = get_saliency_score(llm_generator, context, qa["question"], qa["answer"])
            if "eig" not in qa:
                qa["eig"] = compute_eig_score(llm_generator.sync_client, context, qa["question"], qa["answer"])

        if chapter_qa_pairs:
            chapter_qa_output_file = qa_output_file.replace('.jsonl', f'_{chapter}.jsonl')
            if os.path.exists(chapter_qa_output_file):
                new_file_name = chapter_qa_output_file
                idx = 1 
                while os.path.exists(new_file_name):
                    new_file_name = chapter_qa_output_file.replace('.jsonl', f'_{idx}.jsonl')
                    idx += 1
                write_jsonl(chapter_qa_pairs, new_file_name)
                chapter_qa_output_file = new_file_name
            else:
                write_jsonl(chapter_qa_pairs, chapter_qa_output_file)
            logger.info(f"✅ Wrote generated QA pairs to: {chapter_qa_output_file}")

if __name__ == "__main__":
    main()


