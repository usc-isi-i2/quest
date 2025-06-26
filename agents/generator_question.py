from agents.base import BaseAgent
from utils_llm import LLMMessage


class QuestionGenerator(BaseAgent[tuple[list[str], str]]):
    def __init__(self, generator):
        super().__init__(generator)

    def generate(
        self,
        anchor: str,
        context: str,
        num_questions: int,
        mode: str = "default",  # default | cot | fewshot
        few_shot_candidates: list | None = None,
    ) -> tuple[list[str], str]:
        # Base prompt (for final example)
        if mode == "cot":
            base_prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.

First, think explicitly what information should be given to the student to help them understand the sentence better.
Next, generate a question that helps the student understand the sentence better based on your thoughts.
Output in following JSON format:
{{
"thought": < thought >,
"question": < question >
}}"""
        else:
            base_prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{{
"question": question
}}"""

        # Build few-shot message if applicable
        full_prompt = ""
        if mode == "fewshot" and few_shot_candidates:
            for example in few_shot_candidates:
                few_context = example["context"]
                few_anchor = example["anchor"]
                few_question = example["question"]

                full_prompt += f"""article: {few_context}
Student is currently reading the sentence: {few_anchor}.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{{"question": "{few_question}"}}

"""

        full_prompt += base_prompt

        messages = [LLMMessage(role="user", content=full_prompt)]

        questions = []
        for _ in range(num_questions):
            response = self.generator.generate_json(
                messages=messages,
                temperature=0.5,
            )
            questions.append(response.content["question"])

        return questions, base_prompt
