import copy
from loguru import logger

from agents.base import BaseAgent
from utils_llm import LLMMessage


class Evaluator(BaseAgent[dict[str, dict[str, str]]]):
    def __init__(self, generator):
        super().__init__(generator)

    def generate(self, predictions: dict[str, dict[str, str]], sections: dict[str, dict]) -> dict[str, dict[str, str]]:
        try:
            document = ""
            for _, section_content in sections.items():
                document += f"{section_content['content']}\n"

            teacher_persona = f"""You are a teacher who is evaluating a student's understanding of a document.

Here is the document: {document}"""

            results = {}
            for q_id, instance in predictions.items():
                question = instance["question"]
                answer = instance["answer"]
                prediction = instance["prediction"]
                evaluate_message = [
                    LLMMessage(
                        role="system",
                        content=teacher_persona,
                    ),
                    LLMMessage(
                        role="user",
                        content=f"""Now, determine the correctness of the student's answers to the following question.

question: {question}
ground truth: {answer}
student's answer: {prediction}

Please provide a score between 0 and 1, where 0 indicates the student's answer is completely incorrect, and 1 indicates the student's answer is completely correct.
If ground truth is not provided (e.g., None), determine the correctness of the student's answer based on your own understanding of the document.

Answer in the following JSON format:
{{
    "score": <score>
    "feedback": "<feedback>"
}}""",
                    ),
                ]

                response = self.generator.generate_json(messages=evaluate_message, temperature=0.0, top_p=1.0, seed=42)
                results[q_id] = copy.deepcopy(instance)
                results[q_id]["score"] = response.content["score"]
                results[q_id]["feedback"] = response.content["feedback"]

        except Exception as e:
            logger.error(f"[evaluator] {e}")
            raise ValueError("Action (evaluator) failed.") from e

        return results
