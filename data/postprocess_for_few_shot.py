import json

from utils_file import read_jsonl, write_jsonl


data = read_jsonl("data/data.jsonl")

all_records = []

for instance in data:
    llm_parsing = instance["llm_parsed_results"]
    exam_questions = llm_parsing["questions"]
    used_sections = set()

    for qa in exam_questions.values():
        question = qa["question"]
        relevant_sections = qa.get("relevant_sections", [])

        try:
            relevant_section_first = int(relevant_sections[0])

            if relevant_section_first in used_sections:
                continue  # Skip if we already added a question for this section

            anchor = llm_parsing["sections"][str(relevant_section_first)]["content"]
            context = ""
            for i in range(1, relevant_section_first):
                context += f"\n{llm_parsing['sections'][str(i)]['content']}"

            # Ensure anchor isn't part of the context
            assert anchor not in context

            question_generation_prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{{
    "question": question
}}"""

            record = {
                "subject": instance["subject"],
                "section": relevant_section_first,
                "messages": [
                    {"role": "user", "content": question_generation_prompt},
                    {"role": "assistant", "content": json.dumps({"question": question})},
                ],
            }

            all_records.append(record)
            used_sections.add(relevant_section_first)

        except (KeyError, ValueError, IndexError, AssertionError):
            print(f"Skipped: {relevant_sections}")

# Save to one file
write_jsonl(all_records, "data/data_post_processed.jsonl")
print(f"✅ Written {len(all_records)} unique section-question pairs to data_post_processed.jsonl")
