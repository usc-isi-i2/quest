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
Student is currently reading the section: {anchor}.

First, think explicitly what kind of information should be given to the student to help them understand the section better. 
Next, generate {num_questions} nonoverlapping questions that help the student understand the section better based on your thoughts. The questions should not be directly answerable with information already presented in the article or the current section. At a minimum, the answer to the questions should require paraphrasing information in the current section.
Output in following JSON format:
{{
"thought": < thought >,
"questions": [question1, question2, ...]
}}"""
        else:
            base_prompt = f"""article: {context}
Student is currently reading the section: {anchor}.
Generate {num_questions} nonoverlapping questions that would help the student understand the current section better. The questions should not be directly answerable with information already presented in the article or the current section. At a minimum, the answer to the questions should require paraphrasing information in the current section.
Output in following JSON format:
{{
"questions": [question1, question2, ...]
}}"""

        # Build few-shot message if applicable
        full_prompt = ""
        if mode == "fewshot" and few_shot_candidates:
            for example in few_shot_candidates:
                few_context = example["context"]
                few_anchor = example["anchor"]
                few_question = example["question"]

                full_prompt += f"""article: {few_context}
Student is currently reading the section: {few_anchor}.
Generate a question that helps the student understand the section better.
Output in following JSON format:
{{"question": "{few_question}"}}

"""

        full_prompt += base_prompt

        messages = [LLMMessage(role="user", content=full_prompt)]

        response = self.generator.generate_json(
            messages=messages,
            temperature=1,
        )
        
        questions = response.content["questions"]

        return questions, base_prompt
