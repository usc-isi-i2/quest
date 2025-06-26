import argparse
import json
import os
import time

from openai import OpenAI

from agents import AnswerGenerator, QuestionGenerator, Simulator
from utils_file import read_jsonl, write_json, write_jsonl
from utils_llm import OpenAIGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="QUEST-based rejection sampling and fine-tuning.")

    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18", help="Base model to fine-tune")
    parser.add_argument("--subject", nargs="+", required=True, help="Subject(s) to include or 'all'")
    parser.add_argument("--iterations", type=int, default=1, help="Number of rejection sampling + fine-tune iterations")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Minimum utility score for training data inclusion"
    )
    parser.add_argument("--use_document_for_simulate", action="store_true", help="Use full document in simulator")
    parser.add_argument("--num_questions_per_section", type=int, default=1, help="Questions to generate per section")
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
    print(f"Metadata saved to {metadata_file}")


def main():
    args = parse_args()
    data = read_jsonl("data/data.jsonl")

    if args.subject == ["all"]:
        args.subject = sorted(set(item["subject"] for item in data))
        print(f"Using all subjects: {args.subject}")

    current_model_name = args.model_name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    for iteration in range(args.iterations):
        print(f"\n========== Iteration {iteration + 1} / {args.iterations} ==========")

        qg_llm = OpenAIGenerator(model=current_model_name, oai_api_key=os.getenv("OPENAI_API_KEY", ""))
        question_generator = QuestionGenerator(generator=qg_llm)
        answer_generator = AnswerGenerator(generator=qg_llm)
        simulator = Simulator(generator=qg_llm)

        high_utility_questions = []

        for subject in args.subject:
            subject_data = [d for d in data if d["subject"] == subject]
            subject_data = sorted(subject_data, key=lambda x: x["chapter"])
            train_data = subject_data[:-5]

            for chapter_data in train_data:
                sections = chapter_data["llm_parsed_results"]["sections"]
                exam_questions = chapter_data["llm_parsed_results"]["questions"]
                context, questions_by_section, prompts_by_section = "", {}, {}

                for i in range(1, len(sections) + 1):
                    anchor = sections[str(i)]["content"]
                    questions, prompt = question_generator.generate(
                        anchor=anchor, context=context, num_questions=args.num_questions_per_section
                    )
                    answers = answer_generator.generate(anchor=anchor, context=context, questions=questions)
                    context += f"\n{anchor}"
                    questions_by_section[str(i)] = answers
                    prompts_by_section[str(i)] = prompt

                try:
                    sections_or_empty = sections if args.use_document_for_simulate else {}
                    utilities, _, _ = simulator.generate(
                        eval_questions=exam_questions,
                        sections=sections_or_empty,
                        generated_questions=questions_by_section,
                    )
                except Exception as e:
                    print(f"Simulation error: {e}")
                    continue

                for qid, u in utilities.items():
                    if u["utility"] > args.threshold:
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

        if not high_utility_questions:
            print("No high-utility questions found. Skipping.")
            continue

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
        print(f"Saved training data: {data_filename}")

        print("\nUploading file for fine-tuning...")
        uploaded_file_id = client.files.create(file=open(data_filename, "rb"), purpose="fine-tune").id

        print("Creating fine-tuning job...")
        job_id = client.fine_tuning.jobs.create(training_file=uploaded_file_id, model=current_model_name).id
        print(f"Started fine-tuning job: {job_id}")

        while True:
            status = client.fine_tuning.jobs.retrieve(job_id).status
            print(f"Job {job_id} status: {status}")
            if status in ["succeeded", "failed"]:
                break
            time.sleep(15)

        if status == "succeeded":
            new_model = client.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
            if not new_model:
                print("No fine-tuned model returned. Stopping.")
                break
            print(f"Fine-tuning succeeded: {new_model}")
            metadata_file = f"metadata/train_metadata_{subjects_str}_iter_{iteration}_thresh_{threshold_str}.json"
            write_metadata(iteration, args.subject, args.threshold, current_model_name, new_model, metadata_file)
            current_model_name = new_model
        else:
            print("Fine-tuning failed. Stopping.")
            break

    print("\nAll iterations complete.")


if __name__ == "__main__":
    main()
