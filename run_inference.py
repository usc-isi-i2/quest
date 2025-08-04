import argparse
import os
import random

from agents import AnswerGenerator, QuestionGenerator, Simulator
from utils_file import read_jsonl, write_jsonl
from utils_llm import OpenAIGenerator
from loguru import logger


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
    parser.add_argument("--use_document_for_simulate", action="store_true", help="Use document sections in simulation")
    parser.add_argument("--num_questions_per_section", type=int, default=1, help="Questions to generate per section")
    parser.add_argument("--run_on_train", action="store_true", help="Run on train data, mainly used to produce data for analysis")
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
    out_file = f"output/{args.qg_model_name}/{subject}_{args.mode}_performance_results.jsonl" if not args.run_on_train else f"output/{args.qg_model_name}/{subject}_{args.mode}_performance_results_on-train:{args.run_on_train}.jsonl"
    qa_output_file = f"output/{args.qg_model_name}/{subject}_{args.mode}_qa_pairs.jsonl"
    question_generator_performances = []
    qa_pairs = []
    few_shot_candidates = []

    filtered_data = [item for item in data if item["subject"] == subject]
    if not filtered_data:
        logger.info(f"No data found for subject '{subject}'")
        return

    filtered_data = sorted(filtered_data, key=lambda x: x["chapter"])
    test_filtered_data = filtered_data[-5:]
    train_filtered_data = filtered_data[:-5]

    if args.mode == "fewshot":
        few_shot_raw_examples = []
        for d in train_filtered_data:
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

    if args.run_on_train: 
        # warn against use for fewshot 
        if args.mode == "fewshot": 
            logger.warning("Running on train data with fewshot mode is not recommended. Please use default mode instead.")
        test_filtered_data = train_filtered_data

    for test_data in test_filtered_data:
        chapter = test_data["chapter"]
        llm_parsing = test_data["llm_parsed_results"]
        exam_questions = llm_parsing["questions"]

        context = ""
        questions_by_section = {}

        for i in range(1, len(llm_parsing["sections"]) + 1):
            anchor = llm_parsing["sections"][str(i)]["content"]

            generated_questions, _ = question_generator.generate(
                anchor=anchor,
                context=context,
                num_questions=args.num_questions_per_section,
                mode=args.mode,
                few_shot_candidates=few_shot_candidates if args.mode == "fewshot" else None,
            )
            generated_qa_pairs = answer_generator.generate(
                anchor=anchor, context=context, questions=generated_questions
            )

            context += f"\n{anchor}"
            questions_by_section[str(i)] = generated_qa_pairs

            for qa in generated_qa_pairs:
                qa_pairs.append(
                    {
                        "subject": subject,
                        "chapter": chapter,
                        "section": i,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

        for section_id, questions in questions_by_section.items():
            logger.info(f"\nSection {section_id}:")
            for q in questions:
                logger.info(f"  Question: {q['question']}")
                logger.info(f"  Answer: {q['answer']}")

        if args.use_document_for_simulate:
            _, all_score, baseline_score = simulator.generate(
                eval_questions=exam_questions,
                sections=llm_parsing["sections"],
                generated_questions=questions_by_section,
                test=True,
            )
        else:
            _, all_score, baseline_score = simulator.generate(
                eval_questions=exam_questions,
                sections={},
                generated_questions=questions_by_section,
                test=True,
            )

        question_generator_performances.append(
            {
                "subject": subject,
                "chapter": chapter,
                "all_score": all_score,
                "baseline_score": baseline_score,
                "gain": all_score - baseline_score,
            }
        )
    
    # add average 
    average_performance = {
        "subject": subject,
        "chapter": "average",
        "all_score": sum([p["all_score"] for p in question_generator_performances]) / len(question_generator_performances),
        "baseline_score": sum([p["baseline_score"] for p in question_generator_performances]) / len(question_generator_performances),
        "gain": sum([p["gain"] for p in question_generator_performances]) / len(question_generator_performances),
    }
    question_generator_performances.append(average_performance)

    write_jsonl(question_generator_performances, out_file)
    write_jsonl(qa_pairs, qa_output_file)
    logger.info(f"\n✅ Wrote performance to: {out_file}")
    logger.info(f"✅ Wrote generated QA pairs to: {qa_output_file}")


if __name__ == "__main__":
    main()
