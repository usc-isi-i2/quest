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
        if mode == "cot":
            question_generation_prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.

First, think explicitly what information should be given to the student to help them understand the sentence better.
Next, generate a question that helps the student understand the sentence better based on your thoughts.
Output in following JSON format:
{{
"thought": < thought >,
"question": < question >
}}"""
        else:
            question_generation_prompt = f"""article: {context}
Student is currently reading the sentence: {anchor}.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{{
"question": question
}}"""

        messages = []
        if mode == "fewshot" and few_shot_candidates:
            for candidate in few_shot_candidates:
                for message in candidate["messages"]:
                    messages.append(LLMMessage(role=message["role"], content=message["content"]))
        messages.append(LLMMessage(role="user", content=question_generation_prompt))

        questions = []
        for _ in range(num_questions):
            response = self.generator.generate_json(
                messages=messages,
                temperature=0.5,
            )
            questions.append(response.content["question"])

        return questions, question_generation_prompt
