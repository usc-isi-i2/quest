import argparse
import json
import os
import time

from openai import OpenAI

from utils_file import read_jsonl, write_json, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate questions.")

    parser.add_argument(
        "--model_name", type=str, required=False, help="Model name, e.g. gpt-4o-mini", default="gpt-4o-mini-2024-07-18"
    )

    parser.add_argument("--subject", type=str, required=True, help="Single subject name (e.g., chemistry) or 'all'")

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument(
        "--use_document_for_simulate", action="store_true", help="Flag to use document sections for the simulator"
    )
    parser.add_argument(
        "--num_questions_per_section",
        type=int,
        default=1,
        help="Number of questions to generate per section (default: 1)",
    )

    return parser.parse_args()


def write_metadata(subjects, base_model, fine_tuned_model, metadata_file):
    """Store iteration metadata in a JSON file for reference."""
    metadata = {
        "subjects": subjects,
        "base_model_used": base_model,
        "fine_tuned_model": fine_tuned_model,
    }
    write_json(metadata, metadata_file)
    print(f"Metadata saved to {metadata_file}")


def main():
    args = parse_args()

    # Read the data
    data = read_jsonl("data/data.jsonl")

    # Determine subjects to process
    if args.subject == "all":
        subjects = sorted(set(item["subject"] for item in data))
        print(f"Using all subjects: {subjects}")
    else:
        subjects = [args.subject]
        print(f"Using subject: {subjects[0]}")

    # Keep track of the current model name; we may update it after each fine-tune iteration
    current_model_name = args.model_name
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    direct_train_data = []
    for subject in subjects:
        # Filter for the requested subject
        filtered_data = [item for item in data if item["subject"] == subject]
        if not filtered_data:
            print(f"No data found for subject '{subject}'")
            continue

        # Sort filtered data by chapter
        filtered_data = sorted(filtered_data, key=lambda x: x["chapter"])
        train_filtered_data = filtered_data[:-5]  # Use all but the last 5 entries for training

        # For each data entry matching the subject, process
        for train_data in train_filtered_data:
            llm_parsing = train_data["llm_parsed_results"]
            exam_questions = llm_parsing["questions"]  # Dict[str, Dict]

            anchor = ""
            context = ""

            for i in range(1, len(llm_parsing["sections"]) + 1):
                anchor = llm_parsing["sections"][str(i)]["content"]
                context += f"\n{anchor}"

            prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{{
    \"question\": question
}}"""
            for _, question in exam_questions.items():
                direct_train_data.append(
                    {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": json.dumps({"question": question["question"]})},
                        ]
                    }
                )

    subjects_str = "_".join(args.subject)

    # Save fine-tune data with relevant naming
    fine_tune_data_filename = f"metadata/train_data_{subjects_str}_sft.jsonl"
    write_jsonl(direct_train_data, fine_tune_data_filename)
    print(f"Created fine-tuning data: {fine_tune_data_filename}")

    # Upload file to OpenAI
    print("\nUploading file for fine-tuning...")
    upload_response = client.files.create(file=open(fine_tune_data_filename, "rb"), purpose="fine-tune")
    uploaded_file_id = upload_response.id
    print(f"Uploaded file id: {uploaded_file_id}")

    # Create FineTuningJob
    print("Creating fine-tuning job...")
    job_response = client.fine_tuning.jobs.create(
        training_file=uploaded_file_id,
        model=current_model_name,
    )
    job_id = job_response.id
    print(f"Created fine-tuning job: {job_id}")

    # Wait for fine-tuning
    while True:
        resp = client.fine_tuning.jobs.retrieve(job_id)
        status = resp.status
        print(f"Fine-tuning job {job_id} status: {status}")
        if status in ["succeeded", "failed"]:
            break
        time.sleep(15)

    if status == "succeeded":
        new_model = resp.fine_tuned_model
        print(f"Fine-tuning succeeded. New model: {new_model}")

        # Write metadata
        metadata_file = f"metadata/train_metadata_{subjects_str}_sft.json"
        write_metadata(
            subjects=args.subject,
            base_model=current_model_name,
            fine_tuned_model=new_model,
            metadata_file=metadata_file,
        )
    else:
        print("Fine-tuning failed. Stopping.")


if __name__ == "__main__":
    main()
