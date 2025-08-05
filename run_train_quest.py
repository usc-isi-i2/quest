import argparse
import json
import os
import time
import traceback
from datetime import datetime

from openai import OpenAI

from agents import AnswerGenerator, QuestionGenerator, Simulator
from utils_file import read_jsonl, write_json, write_jsonl
from utils_llm import OpenAIGenerator
from loguru import logger
from tracking_logger import TrackingLogger


def parse_args():
    parser = argparse.ArgumentParser(description="QUEST-based rejection sampling and fine-tuning.")

    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune")
    parser.add_argument("--subject", type=str, required=True, help="Subject to include (e.g., 'chemistry') or 'all'")
    parser.add_argument("--iterations", type=int, default=1, help="Number of rejection sampling + fine-tune iterations")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Minimum utility score for training data inclusion"
    )
    parser.add_argument("--use_document_for_simulate", action="store_true", help="Use full document in simulator")
    parser.add_argument("--num_questions_per_section", type=int, default=1, help="Questions to generate per section")
    parser.add_argument("--tracking_file", type=str, default=None, help="JSONL file to track intermediate outputs")
    parser.add_argument("--test_mode", action="store_true", help="Run on small subset for testing logging functionality")
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


def main():
    args = parse_args()
    data = read_jsonl("data/data.jsonl")

    # Setup overall tracking file
    if args.tracking_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.tracking_file = f"tracking_outputs_{args.subject}_{timestamp}.jsonl"
    
    logger.info(f"Tracking intermediate outputs to: {args.tracking_file}")

    # subject handling
    if args.subject == "all":
        subjects = sorted(set(item["subject"] for item in data))
        logger.info(f"Using all subjects: {subjects}")
    else:
        subjects = [args.subject]
        logger.info(f"Using subject: {args.subject}")
    
    # Test mode: limit to small subset
    if args.test_mode:
        logger.info("TEST MODE: Running on small subset for logging verification")
        # Limit to first subject only
        subjects = subjects[:1]
        # Limit iterations to 1
        args.iterations = 1
        # Lower threshold for test mode to see generated questions
        args.threshold = 0.0
        # Limit chapters per subject to 1
        logger.info(f"Test mode: Using {subjects[0]} with 1 iteration and 1 chapter")
        logger.info(f"Test mode: Lowered threshold to {args.threshold} to see all generated questions")

    current_model_name = args.model_name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    for iteration in range(args.iterations):
        logger.info(f"\n========== Iteration {iteration + 1} / {args.iterations} ==========")
        
        # Initialize tracking logger for this iteration
        tracker = TrackingLogger(args.tracking_file, iteration=iteration + 1)
        
        # Log iteration start
        tracker.log_iteration_start(
            total_iterations=args.iterations,
            model_name=current_model_name,
            subjects=subjects,
            threshold=args.threshold,
            use_document_for_simulate=args.use_document_for_simulate,
            num_questions_per_section=args.num_questions_per_section
        )

        qg_llm = OpenAIGenerator(model=current_model_name, oai_api_key=os.getenv("OPENAI_API_KEY", ""))
        question_generator = QuestionGenerator(generator=qg_llm)
        answer_generator = AnswerGenerator(generator=qg_llm)
        simulator = Simulator(generator=qg_llm)

        high_utility_questions = []

        for subject in subjects:
            subject_data = [d for d in data if d["subject"] == subject]
            subject_data = sorted(subject_data, key=lambda x: x["chapter"])
            train_data = subject_data[:-5]
            
            # Test mode: limit to first 3 chapters only
            if args.test_mode:
                train_data = train_data[:3]
                logger.info(f"Test mode: Processing only first 3 chapters: {train_data[0]['chapter']}")
                logger.info(f"Test mode: Found {len(train_data)} chapters to process")
                logger.info(f"Test mode: Chapter data keys: {list(train_data[0].keys())}")
                logger.info(f"Test mode: Sections available: {list(train_data[0]['llm_parsed_results']['sections'].keys())}")
                logger.info(f"Test mode: Questions available: {list(train_data[0]['llm_parsed_results']['questions'].keys())}")

            # Set subject and log subject processing start
            tracker.set_subject(subject)
            tracker.log_subject_start(total_chapters=len(train_data))

            for chapter_idx, chapter_data in enumerate(train_data):
                chapter_id = chapter_data["chapter"]
                sections = chapter_data["llm_parsed_results"]["sections"]
                exam_questions = chapter_data["llm_parsed_results"]["questions"]
                context, questions_by_section, prompts_by_section = "", {}, {}
                
                                # Create chapter-specific log file
                chapter_log_file = f"logs/chapter_{subject}_{chapter_id}_iter_{iteration + 1}.jsonl"
                logger.info(f"Creating chapter log file: {chapter_log_file}")
                
                # Create chapter-specific tracker
                chapter_tracker = TrackingLogger(args.tracking_file, chapter_log_file, 
                                              iteration=iteration + 1, subject=subject, chapter_id=chapter_id)
                
                # Log chapter processing start
                chapter_tracker.log_chapter_start(
                    chapter_index=chapter_idx,
                    total_sections=len(sections),
                    total_exam_questions=len(exam_questions)
                )

                for i in range(1, len(sections) + 1):                 
                    anchor = sections[str(i)]["content"]
                    logger.info(f"Test mode: Processing section {i}, anchor length: {len(anchor)}")
                    
                    # Test mode: limit to first 3 sections only
                    if args.test_mode and i > 3:
                        logger.info(f"Test mode: Skipping section {i} due to test mode limit")
                        break
                    
                    questions, prompt = question_generator.generate(
                        anchor=anchor, context=context, num_questions=args.num_questions_per_section
                    )
                    logger.info(f"Test mode: Generated {len(questions)} questions for section {i}")
                    
                    answers = answer_generator.generate(anchor=anchor, context=context, questions=questions)
                    logger.info(f"Test mode: Generated {len(answers)} answers for section {i}")
                    
                    context += f"\n{anchor}"
                    questions_by_section[str(i)] = answers
                    prompts_by_section[str(i)] = prompt

                    # Log question generation for this section
                    chapter_tracker.log_section_questions(
                        section_id=str(i),
                        anchor=anchor,
                        questions=questions,
                        answers=answers,
                        prompt=prompt
                    )

                try:
                    sections_or_empty = sections if args.use_document_for_simulate else {}
                    logger.info(f"Test mode: Starting simulation with {len(exam_questions)} exam questions")
                    logger.info(f"Test mode: Generated questions by section: {list(questions_by_section.keys())}")
                    logger.info(f"Test mode: Total generated questions: {sum(len(qs) for qs in questions_by_section.values())}")
                    
                    utilities, all_score, baseline_score, individual_scores = simulator.generate(
                        eval_questions=exam_questions,
                        sections=sections_or_empty,
                        generated_questions=questions_by_section,
                    )
                    
                    logger.info(f"Test mode: Simulation completed, found {len(utilities)} utilities")
                    
                    # Log simulation results (includes individual scores)
                    chapter_tracker.log_simulation_results(
                        baseline_score=baseline_score,
                        all_questions_score=all_score,
                        total_generated_questions=len(utilities),
                        exam_questions=exam_questions,
                        utilities=utilities,
                        individual_scores=individual_scores
                    )
                    
                    # Log detailed tracking for each generated question
                    for qid, utility_info in utilities.items():
                        generated_question_detail = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "generated_question_detail",
                            "iteration": iteration + 1,
                            "subject": subject,
                            "chapter_id": chapter_id,
                            "generated_question_id": qid,
                            "question": utility_info["question"],
                            "answer": utility_info["answer"],
                            "section": utility_info["section"],
                            "baseline_score": baseline_score,
                            "single_gain": utility_info["single_gain"],
                            "all_questions_score": all_score,
                            "all_but_one_gain": utility_info["all_but_one_gain"],
                            "utility": utility_info["utility"],
                            "exam_questions": {}
                        }
                        
                        # Add exam question details for this generated question to see how each generated question affects performance on individual exam questions
                        for exam_qid, exam_question in exam_questions.items():
                            exam_detail = {
                                "question": exam_question["question"],
                                "ground_truth": exam_question["answer"],
                                "baseline": {
                                    "prediction": individual_scores["baseline"][exam_qid]["prediction"],
                                    "score": individual_scores["baseline"][exam_qid]["score"]
                                },
                                "all_questions": {
                                    "prediction": individual_scores["all_questions"][exam_qid]["prediction"],
                                    "score": individual_scores["all_questions"][exam_qid]["score"]
                                },
                                "single_one": {
                                    "prediction": individual_scores["single_questions"][qid][exam_qid]["prediction"],
                                    "score": individual_scores["single_questions"][qid][exam_qid]["score"]
                                },
                                "leave_one_out": {
                                    "prediction": individual_scores["leave_one_out"][qid][exam_qid]["prediction"],
                                    "score": individual_scores["leave_one_out"][qid][exam_qid]["score"]
                                }
                            }
                            generated_question_detail["exam_questions"][exam_qid] = exam_detail
                        
                        chapter_tracker.log_generated_question_detail(
                            qid=qid,
                            question=utility_info["question"],
                            answer=utility_info["answer"],
                            section=utility_info["section"],
                            single_gain=float(utility_info["single_gain"]),
                            all_but_one_gain=float(utility_info["all_but_one_gain"]),
                            utility=float(utility_info["utility"]),
                            exam_questions=generated_question_detail["exam_questions"]
                        )

                    for qid, u in utilities.items():
                        if float(u["utility"]) > args.threshold:
                            high_utility_questions.append(
                                {
                                    "qid": qid,
                                    "question": u["question"],
                                    "answer": u["answer"],
                                    "section": u["section"],
                                    "utility": u["utility"],
                                    "prompt": prompts_by_section[u["section"]],
                                }
                            )

                except Exception as e:
                    logger.info(f"Simulation error: {e}")
                    # Log simulation error, stack trace and quit
                    chapter_tracker.log_simulation_error(error=str(e))
                    logger.info(f"Stack trace: {traceback.format_exc()}")
                    return 1

                # Log chapter completion
                chapter_tracker.log_chapter_complete(
                    high_utility_count=len([q for q in high_utility_questions if q.get("chapter_id") == chapter_id])
                )

            # Log subject completion
            tracker.log_subject_complete(
                total_high_utility=len([q for q in high_utility_questions if q.get("subject") == subject])
            )

        if not high_utility_questions:
            logger.info("No high-utility questions found. Skipping.")
            # Log no high utility questions found
            tracker.log_no_high_utility_questions(threshold=args.threshold)
            continue

        # Test mode: skip fine-tuning
        if args.test_mode:
            logger.info("TEST MODE: Skipping fine-tuning for testing purposes")
            tracker.log_test_mode_complete(
                total_high_utility_questions=len(high_utility_questions)
            )
            continue

        # Log high utility questions summary
        tracker.log_high_utility_questions_summary(high_utility_questions=high_utility_questions)

        subjects_str = "_".join(args.subject)
        threshold_str = str(args.threshold).replace(".", "p")
        data_filename = f"metadata/train_data_{subjects_str}_iter_{iteration}_thresh_{threshold_str}.jsonl"

        training_data = [
            {
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": json.dumps({"question": item["question"]})},
                ]
            }
            for item in high_utility_questions
        ]

        write_jsonl(training_data, data_filename)
        logger.info(f"Saved training data: {data_filename}")

        # Log training data creation
        tracker.log_training_data_created(
            training_data_file=data_filename,
            training_data_count=len(training_data)
        )

        logger.info("\nUploading file for fine-tuning...")
        uploaded_file_id = client.files.create(file=open(data_filename, "rb"), purpose="fine-tune").id

        logger.info("Creating fine-tuning job...")
        job_id = client.fine_tuning.jobs.create(training_file=uploaded_file_id, model=current_model_name).id
        logger.info(f"Started fine-tuning job: {job_id}")

        # Log fine-tuning job start
        tracker.log_fine_tuning_job_start(
            job_id=job_id,
            model_name=current_model_name,
            training_file_id=uploaded_file_id
        )

        while True:
            status = client.fine_tuning.jobs.retrieve(job_id).status
            logger.info(f"Job {job_id} status: {status}")
            
            # Log fine-tuning status update
            tracker.log_fine_tuning_status(
                job_id=job_id,
                status=status
            )
            
            if status in ["succeeded", "failed"]:
                break
            time.sleep(15)

        if status == "succeeded":
            new_model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
            if not new_model:
                logger.info("No fine-tuned model returned. Stopping.")
                # Log fine-tuning failure
                tracker.log_fine_tuning_no_model(job_id=job_id)
                break
            logger.info(f"Fine-tuning succeeded: {new_model}")
            metadata_file = f"metadata/train_metadata_{subjects_str}_iter_{iteration}_thresh_{threshold_str}.json"
            write_metadata(iteration, args.subject, args.threshold, current_model_name, new_model, metadata_file)
            current_model_name = new_model
            
            # Log fine-tuning success
            tracker.log_fine_tuning_success(
                job_id=job_id,
                old_model=current_model_name,
                new_model=new_model,
                metadata_file=metadata_file
            )
        else:
            logger.info("Fine-tuning failed. Stopping.")
            # Log fine-tuning failure
            tracker.log_fine_tuning_failed(job_id=job_id)
            break

        # Log iteration completion
        tracker.log_iteration_complete(final_model=current_model_name)

    logger.info("\nAll iterations complete.")

if __name__ == "__main__":
    main()
