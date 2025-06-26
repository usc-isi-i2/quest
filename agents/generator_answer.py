from agents.base import BaseAgent
from utils_llm import LLMMessage


class AnswerGenerator(BaseAgent[list[dict[str, str]]]):
    def __init__(self, generator):
        super().__init__(generator)

    def generate(self, anchor: str, context: str, questions: list[str]) -> list[dict[str, str]]:
        qa_generation_prompt = f"""questions: {questions}

Answer each question shortly and output in following JSON format:
{{
"qa_pairs": [
    {{"question": question_1, "answer": answer_1}},
    {{"question": question_2, "answer": answer_2}},
    ...
    {{"question": question_n, "answer": answer_n}},
]
}}"""
        qa_generation_messages = [
            LLMMessage(
                role="user",
                content=qa_generation_prompt,
            ),
        ]

        response = self.generator.generate_json(messages=qa_generation_messages, temperature=0.0, top_p=1.0, seed=42)

        qa_pairs = response.content["qa_pairs"][: len(questions)]

        return qa_pairs
