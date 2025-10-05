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
from run_eval import compute_saliency_with_fallback, compute_eig_score
from functools import partial
from tqdm import tqdm
from collections import defaultdict
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

def score_questions(questions_by_section, scoring_function, threshold, prompts_by_section):
    high_metric_questions = []
    for section_id, questions in questions_by_section.items():
        for q_idx, question in enumerate(questions):
            article = question["context"] + question["anchor"]
            metric = scoring_function(article, question["question"], question["answer"])
            if metric > threshold:
                high_metric_questions.append(
                    {
                        "qid": f"{section_id}_Q{q_idx+1}",
                        "question": question["question"],
                        "answer": question["answer"],
                        "section": section_id,
                        "metric": metric,
                        "prompt": prompts_by_section[section_id],
                    }
                )
    return high_metric_questions
                
                    


def main():
    args = parse_args()
    
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

    if args.thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    else:
        if len(args.metrics) != len(args.thresholds):
            raise ValueError(f"Number of metrics and thresholds must be the same: {args.metrics} and {args.thresholds}")
        thresholds = {metric: float(threshold) for metric, threshold in zip(args.metrics, args.thresholds)}
        
    if len(args.metrics) != 1 and args.iterations > 1:
        raise ValueError("Only one metric can be used for training data selection when running multiple iterations")
        
    logger.info(f"Using thresholds: {thresholds}")
    if "saliency" in args.metrics:
        logger.info("Loading QSalience model (this may take 30-60 seconds)...")
        from utils_qsalience import QSalience
        qsalience = QSalience()
        logger.info("✅ QSalience model loaded successfully")
        
    data = read_jsonl("data/data.jsonl")
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
                train_data = train_data[:1]

            for chapter_data in tqdm(train_data, desc="Processing chapters"):
                sections = chapter_data["llm_parsed_results"]["sections"]
                exam_questions = chapter_data["llm_parsed_results"]["questions"]
                context, questions_by_section, prompts_by_section = "", {}, {}

                for i in range(1, len(sections) + 1):                    
                    if args.test and i > 3:
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

                if "utility" in args.metrics:
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
                        if u["utility"] > thresholds["utility"]:
                            high_metric_questions["utility"].append(
                                {
                                    "qid": qid,
                                    "question": u["question"],
                                    "answer": u["answer"],
                                    "section": u["section"],
                                    "metric": u["utility"],
                                    "prompt": prompts_by_section[u["section"]],
                                }
                            )
                if "saliency" in args.metrics:
                    sal_llm = OpenAIGenerator(
                        model=current_model_name, 
                        oai_api_key=os.getenv("OPENAI_API_KEY", ""),
                        operation_type="saliency_scoring"
                    )
                    scoring_function = partial(compute_saliency_with_fallback, qsalience, sal_llm)
                    high_metric_questions["saliency"] = score_questions(questions_by_section, scoring_function, thresholds["saliency"], prompts_by_section)
                
                if "eig" in args.metrics:
                    eig_llm = OpenAIGenerator(
                        model=current_model_name, 
                        oai_api_key=os.getenv("OPENAI_API_KEY", ""),
                        operation_type="eig_scoring"
                    )
                    scoring_function = partial(compute_eig_score, eig_llm.sync_client)
                    high_metric_questions["eig"] = score_questions(questions_by_section, scoring_function, thresholds["eig"], prompts_by_section)

        training_data_paths = {}
        for metric, questions in high_metric_questions.items():
            if len(questions) == 0:
                logger.info(f"No high-{metric} questions found. Skipping.")
                continue
            else: 
                logger.info(f"Found {len(questions)} high-{metric} questions to use for fine-tuning.")

            subjects_str = "_".join(args.subject)
            threshold_str = str(thresholds[metric]).replace(".", "p")
            data_filename = f"metadata/train_data_{metric}_{subjects_str}_iter_{iteration}_thresh_{threshold_str}_n{args.num_questions_per_section}.jsonl"
            if args.test:
                data_filename = data_filename.replace(".jsonl", "_test.jsonl")
                
            training_data = [
                {
                    "messages": [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": json.dumps({"question": item["question"]})},
                    ]
                }
                for item in high_metric_questions
            ]

            write_jsonl(training_data, data_filename)
            logger.info(f"Saved training data: {data_filename}")
            training_data_paths[metric] = data_filename
            
        fine_tuning_job_ids = {}
        for metric, data_filename in training_data_paths.items():
            logger.info("\nUploading file for fine-tuning...")
            uploaded_file_id = client.files.create(file=open(data_filename, "rb"), purpose="fine-tune").id

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

            if status == "succeeded":
                new_model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
                if not new_model:
                    logger.info("No fine-tuned model returned. Stopping.")
                    break
                logger.info(f"Fine-tuning succeeded: {new_model}")
                metadata_file = data_filename.replace("data", "metadata")
                write_metadata(iteration, args.subject, thresholds[metric], current_model_name, new_model, metadata_file)
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
