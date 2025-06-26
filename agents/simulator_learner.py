from loguru import logger

from agents.base import BaseAgent
from utils_llm import LLMMessage


class Learner(BaseAgent[dict[str, dict[str, str]]]):
    def __init__(self, generator):
        super().__init__(generator)

    def generate(
        self,
        eval_questions: dict[str, dict],
        sections: dict[str, dict],
        generated_questions: dict[str, list],
    ) -> dict[str, dict[str, str]]:
        try:
            learning_material = ""
            if not sections:
                for _, questions in generated_questions.items():
                    for _, question in enumerate(questions, start=1):
                        learning_material += (
                            f"Practice question: {question['question']}\nanswer: {question['answer']}\n\n"
                        )
            else:
                for section_id, section_content in sections.items():
                    learning_material += f"section {section_id}: {section_content['content']}\n\n"
                    if section_id in generated_questions:
                        for _, question in enumerate(generated_questions[section_id], start=1):
                            learning_material += (
                                f"Practice question: {question['question']}\nanswer: {question['answer']}\n\n"
                            )

            exam = ""
            for q_id, question in eval_questions.items():
                exam += f"Exam question {q_id}: {question['question']}\n\n"

            prompt = f"""You are now a learner participating in a structured learning simulation. Your task is to:

1. **Study the Provided Learning Materials:**
   Carefully read and understand the content enclosed in the [LEARNING MATERIALS] tags.

2. **Answer the Exam Questions Using Only the Learning Materials:**
   When you respond to the questions in the [EXAM], you must:
   - Base all answers solely on the information contained in the [LEARNING MATERIALS].
   - Clearly show how your reasoning follows from the [LEARNING MATERIALS].
   - If the question asks about something not covered in the [LEARNING MATERIALS], do not provide an answer or guess. Instead, respond exactly with:
     ```
     I don’t know. I have not been studied on this.
     ```
   - Do not use information from outside the [LEARNING MATERIALS].

3. **No External Knowledge or Guessing:**
   Provide no additional reasoning or information if the content is not in the [LEARNING MATERIALS].

Let’s begin.

[LEARNING MATERIALS]
{learning_material}
[/LEARNING MATERIALS]

Now, proceed to the exam below and answer as instructed:

[EXAM]
{exam}
[/EXAM]

Response answers in following JSON format (key: Exam question number, value: your answer):
{{
    "1": < your answer to exam question 1 >,
    "2": < your answer to exam question 2 >,
    ...
}}
"""
            simulation_messages = [
                LLMMessage(
                    role="user",
                    content=prompt,
                ),
            ]

            response = self.generator.generate_json(messages=simulation_messages, temperature=0.0, top_p=1.0, seed=42)

            for q_id, answer in response.content.items():
                if q_id in eval_questions:
                    eval_questions[q_id]["prediction"] = answer
                else:
                    eval_questions.pop(q_id, None)  # Safe deletion without KeyError

        except Exception as e:
            logger.error(f"[learner] {e}")
            raise ValueError("Action (learner) failed.") from e
        return eval_questions
