import argparse
import json
import os
import time
from datetime import datetime

from openai import OpenAI

from agents import AnswerGenerator, QuestionGenerator, Simulator
from utils_file import read_jsonl, write_json, write_jsonl
from utils_llm import OpenAIGenerator
from utils_cost_tracking import get_cost_tracker, reset_cost_tracker
from loguru import logger
from run_eval import compute_saliency_with_fallback, compute_eig_score, build_article_context
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

DEFAULT_THRESHOLDS = {
    "utility": 0.1,
    "saliency": 4,
    "eig": 0,
}

def parse_args():
    parser = argparse.ArgumentParser(description="QUEST-based rejection sampling and fine-tuning.")

    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune")
    parser.add_argument("--subject", type=str, required=True, help="Subject to include (e.g., 'chemistry') or 'all'")
    parser.add_argument("--iterations", type=int, default=1, help="Number of rejection sampling + fine-tune iterations")
    parser.add_argument("--metrics", nargs="+", default=["utility"], choices=["utility", "saliency", "eig"], help="Metrics to use for training data selection, subset of {utility, saliency, eig}")
    parser.add_argument(
        "--thresholds", nargs="+", default=None, help="Minimum metric score for training data inclusion, defaults to 0.1, 4, and 0 for utility, saliency, and eig respectively"
    )
    parser.add_argument("--single_only_utility", action="store_true", help="Only use single-only utility for training data selection")
    parser.add_argument("--use_document_for_simulate", action="store_true", help="Use full document in simulator")
    parser.add_argument("--num_questions_per_section", type=int, default=1, help="Questions to generate per section")
    parser.add_argument("--test", action="store_true", help="Test mode")
    return parser.parse_args()


def write_metadata(iteration, subjects, threshold, base_model, fine_tuned_model, metadata_file):
    metadata = {
        "iteration": iteration,
        "subjects": subjects,
        "threshold": threshold,
        "base_model_used": base_model,
        "fine_tuned_model": fine_tuned_model,
    }
    write_json(metadata, metadata_file)
    logger.info(f"Metadata saved to {metadata_file}")

def score_questions(questions, scoring_function, metric_type):
    for q in questions: 
        if metric_type not in q: 
            article = q["context"] + q["anchor"]
            metric = scoring_function(article, q["question"], q["answer"])
            q[metric_type] = metric


def append_jsonl(new_data: list[dict], data_filename: str):
    with open(data_filename, "a") as f:
        for q in new_data:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

def append_generated_data(questions: list[dict], data_filename: str):
    """Append generated data to file."""
    if len(questions) == 0:
        return
    
    append_jsonl(questions, data_filename)
    logger.info(f"Appended {len(questions)} generated questions to {data_filename}")

def form_trainig_data(questions, prompt_template):
    """Form training data for OpenAI API from questions."""
    if len(questions) == 0:
        return
    
    training_data = [
        {
            "messages": [
                {"role": "user", "content": prompt_template.format(context=item["context"], anchor=item["anchor"], n_questions=1)},
                {"role": "assistant", "content": json.dumps({"questions": [item["question"]]})},
            ]
        }
        for item in questions
    ]
    
    return training_data 
                
def main():
    args = parse_args()
    data = read_jsonl("data/data.jsonl")
    
    # Initialize cost tracking
    reset_cost_tracker()
    tracker = get_cost_tracker()
    logger.info("Cost tracking initialized")

    # subject handling
    if args.subject == "all":
        subjects = sorted(set(item["subject"] for item in data))
        logger.info(f"Using all subjects: {subjects}")
    else:
        subjects = [args.subject]
        logger.info(f"Using subject: {args.subject}")
    subjects_str = "_".join(subjects)

    if args.thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    else:
        if len(args.metrics) != len(args.thresholds):
            raise ValueError(f"Number of metrics and thresholds must be the same: {args.metrics} and {args.thresholds}")
        thresholds = {metric: float(threshold) for metric, threshold in zip(args.metrics, args.thresholds)}
        
    if len(args.metrics) != 1 and args.iterations > 1:
        raise ValueError("Only one metric can be used for training data selection when running multiple iterations")
        
    logger.info(f"Using thresholds: {thresholds}")
    qsalience = None
    if "saliency" in args.metrics:
        logger.info("Loading QSalience model (this may take 30-60 seconds)...")
        from utils_qsalience import QSalience
        qsalience = QSalience()
        logger.info("✅ QSalience model loaded successfully")
        
    generated_questions_data_filename = f"metadata/generated_questions.jsonl"
    generated_questions_data = [] if not os.path.exists(generated_questions_data_filename) else read_jsonl(generated_questions_data_filename)
    generated_question_df = pd.DataFrame(generated_questions_data)

    current_model_name = args.model_name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    for iteration in range(args.iterations):
        logger.info(f"\n========== Iteration {iteration + 1} / {args.iterations} ==========")

        qg_llm = OpenAIGenerator(
            model=current_model_name, 
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
            operation_type="question_generation"
        )
        question_generator = QuestionGenerator(generator=qg_llm)
        
        ag_llm = OpenAIGenerator(
            model=current_model_name, 
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
            operation_type="answer_generation"
        )
        answer_generator = AnswerGenerator(generator=ag_llm)
        
        sim_llm = OpenAIGenerator(
            model=current_model_name, 
            oai_api_key=os.getenv("OPENAI_API_KEY", ""),
            operation_type="simulation"
        )
        simulator = Simulator(generator=sim_llm)

        high_metric_questions = defaultdict(list)
        
        for subject in subjects:
            subject_data = [d for d in data if d["subject"] == subject]
            subject_data = sorted(subject_data, key=lambda x: x["chapter"])
            train_data = subject_data[:-5]
 
            if args.test:
                train_data = train_data[:2]
                args.single_only_utility = True

            for chapter_data in tqdm(train_data, desc="Processing chapters"):
                chapter_id = chapter_data["chapter"]
                logger.info(f"Processing chapter {chapter_id}")
                
                sections = chapter_data["llm_parsed_results"]["sections"]
                exam_questions = chapter_data["llm_parsed_results"]["questions"]
                context, questions_by_section, prompts_by_section = "", {}, {}

                for i in range(1, len(sections) + 1):                    
                    if args.test and i > 1:
                        continue 
                    
                    # if # questions in generated_questions_data is greater than args.num_questions_per_section for this chapter and section, skip 
                    if not generated_question_df.empty and len(generated_question_df[(generated_question_df["chapter"] == chapter_id) & (generated_question_df["section"] == str(i))]) >= args.num_questions_per_section:
                        continue
                                        
                    anchor = sections[str(i)]["content"]
                    questions, prompt = question_generator.generate(
                        anchor=anchor, context=context, num_questions=args.num_questions_per_section
                    )
                    answers = answer_generator.generate(anchor=anchor, context=context, questions=questions)
                    for answer in answers:
                        answer["context"] = context
                        answer['anchor'] = anchor
                    context += f"\n{anchor}"
                    questions_by_section[str(i)] = answers
                    prompts_by_section[str(i)] = prompt

                # Process each metric and append data immediately
                chapter_questions = []
                                
                if "utility" in args.metrics and questions_by_section:
                    try:
                        sections_or_empty = sections if args.use_document_for_simulate else {}
                        utilities, _, _ = simulator.generate(
                            eval_questions=exam_questions,
                            sections=sections_or_empty,
                            generated_questions=questions_by_section,
                            single_only=args.single_only_utility,
                        )
                    except Exception as e:
                        logger.info(f"Simulation error: {e}")
                        continue

                    for qid, u in utilities.items():
                        
                        chapter_questions.append(
                            {
                                "qid": qid,
                                "question": u["question"],
                                "answer": u["answer"],
                                "subject": subject,
                                "chapter": chapter_id,
                                "section": u["section"],
                                "utility": u["utility"],
                                "context": "".join([sections[str(i)]["content"] for i in range(1, int(u["section"]))]),
                                "anchor": sections[str(u["section"])]["content"],
                                "iteration": iteration,
                            }
                        )
                
                # Load existing questions for this chapter, which may not have eig and saliency scores 
                if not generated_question_df.empty:
                    existing_chapter_questions = generated_question_df[
                        (generated_question_df["chapter"] == chapter_id)
                    ].to_dict(orient="records")
                    chapter_questions.extend(existing_chapter_questions)
                    logger.info(f"Loaded {len(existing_chapter_questions)} existing questions for chapter {chapter_id}")
                else:
                    logger.info(f"No existing questions found for chapter {chapter_id}")
                
                if "saliency" in args.metrics and qsalience is not None:
                    sal_llm = OpenAIGenerator(
                        model=current_model_name, 
                        oai_api_key=os.getenv("OPENAI_API_KEY", ""),
                        operation_type="saliency_scoring"
                    )
                    scoring_function = partial(compute_saliency_with_fallback, qsalience, sal_llm)
                    score_questions(chapter_questions, scoring_function, "saliency")
                
                if "eig" in args.metrics:
                    eig_llm = OpenAIGenerator(
                        model=current_model_name, 
                        oai_api_key=os.getenv("OPENAI_API_KEY", ""),
                        operation_type="eig_scoring"
                    )
                    scoring_function = partial(compute_eig_score, eig_llm.sync_client)
                    score_questions(chapter_questions, scoring_function, "eig")
                
                # Append all questions after processing this chapter
                append_generated_data(chapter_questions, generated_questions_data_filename)

        if args.test: 
            logger.info("Test mode. Stopping.")
            break
            
        fine_tuning_job_ids = {}
        training_data_files = {}
        generated_questions_data = read_jsonl(generated_questions_data_filename)
        
        
        for metric in args.metrics: 
            # Filter questions that meet the threshold for this metric
            if args.test: 
                filtered_questions = [q for q in generated_questions_data if metric in q and q["subject"] in subjects][:2]
            else: 
                filtered_questions = [q for q in generated_questions_data if metric in q and q[metric] > thresholds[metric] and q["subject"] in subjects]
            
            logger.info(f"Filtered {len(filtered_questions)} out of {len(generated_questions_data)} {metric} questions (threshold: {thresholds[metric]}, subjects: {subjects})")
            
            if len(filtered_questions) == 0:
                logger.info(f"No {metric} questions meet the threshold. Skipping fine-tuning for {metric}.")
                continue
            
            # Convert filtered data to training format
            training_data = form_trainig_data(filtered_questions, question_generator.base_prompt)
            
            # Write training data to a temporary file
            finetuning_data_name = f"metadata/training_data_{subjects_str}_{metric}_threshold{thresholds[metric]}_iter_{iteration}.jsonl"
            training_data_files[metric] = finetuning_data_name
            write_jsonl(training_data, finetuning_data_name)
            
            if args.test:
                logger.info("Test mode. Skipping fine-tuning.")
                continue 
            
            logger.info("Uploading file for fine-tuning...")
            uploaded_file_id = client.files.create(file=open(finetuning_data_name, "rb"), purpose="fine-tune").id

            logger.info("Creating fine-tuning job...")
            job_id = client.fine_tuning.jobs.create(training_file=uploaded_file_id, model=current_model_name).id
            logger.info(f"Started fine-tuning job: {job_id}")
            fine_tuning_job_ids[metric] = job_id
            
        finetuning_failed = False 
        for metric, job_id in fine_tuning_job_ids.items():
            while True:
                status = client.fine_tuning.jobs.retrieve(job_id).status
                logger.info(f"Job {job_id} status: {status}")
                if status in ["succeeded", "failed"]:
                    break
                time.sleep(15)

            data_filename = training_data_files[metric]
            if status == "succeeded":
                new_model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
                if not new_model:
                    logger.info("No fine-tuned model returned. Stopping.")
                    break
                logger.info(f"Fine-tuning succeeded: {new_model}")
                metadata_file = data_filename.replace("_data_", "_metadata_")
                write_metadata(iteration, args.subject, thresholds[metric], current_model_name, new_model, metadata_file)
                if args.iterations > 1: 
                    current_model_name = new_model
            else:
                logger.info("Fine-tuning failed. Stopping.")
                finetuning_failed = True
                break 
            
        if finetuning_failed:
            break

    logger.info("\nAll iterations complete.")
    
    # Print and save cost summary
    logger.info("\n" + "="*60)
    logger.info("FINAL COST SUMMARY")
    logger.info("="*60)
    tracker.print_summary()
    
    # Save detailed cost log
    cost_log_file = tracker.save_log()
    logger.info(f"Detailed cost log saved to: {cost_log_file}")
    
    # Save cost summary to JSON
    summary = tracker.get_summary()
    summary_file = f"cost_summary_{args.subject}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_tokens": summary.total_tokens,
            "total_cost_usd": summary.total_cost_usd,
            "call_count": summary.call_count,
            "calls_by_operation": summary.calls_by_operation,
            "calls_by_model": summary.calls_by_model,
            "subject": args.subject,
            "iterations": args.iterations,
            "metrics": args.metrics,
            "thresholds": thresholds
        }, f, indent=2)
    logger.info(f"Cost summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
