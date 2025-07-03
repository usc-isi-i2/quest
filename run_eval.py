import argparse
import os

from loguru import logger
import numpy as np

from agents import Simulator
from utils_file import read_jsonl, write_jsonl
from utils_llm import LLMMessage, OpenAIGenerator


textbook_data = read_jsonl("data/data.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate question quality metrics.")
    parser.add_argument("--qa_file", type=str, required=True, help="Path to qa_pairs.jsonl file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--output_dir", type=str, default="q_metrics", help="Directory to save results.")
    parser.add_argument("--include_saliency", action="store_true", help="Whether to compute saliency score.")
    parser.add_argument("--include_eig", action="store_true", help="Whether to compute expected information gain.")
    return parser.parse_args()


def get_saliency_score(llm, article, question, answer):
    try:
        saliency_prompt = (
            f"article: {article}\n"
            f"question: {question}\n"
            "system: Imagine you are a curious reader who is reading the article. "
            "You come across a question and need to determine if it should be answered "
            "in the following article or not.\n"
            "You have to give a score for this input.\n"
            "Score = 1: The question is completely unrelated to the topic of the article.\n"
            "Score = 2: The question is related but has already mostly been answered by the article.\n"
            "Score = 3: The question is related but answering it is not useful as it expands on an idea that is not very important or central.\n"
            "Score = 4: The question is related and answering it enhances the understanding of the article.\n"
            "Score = 5: The question is related and should definitely be answered because it expands on ideas central to the article.\n"
            "Note: The score is based on the information utility of its answer.\n"
            "- If a question is related but does not need to be answered or is not central, do NOT give it a high score.\n"
            "- Score 3 if the question is unanswered but not critical, 2 if it has already been answered.\n"
            "- To differentiate between scores 4 and 5: If the article would feel incomplete without the answer, give 5; otherwise, give 4.\n"
            "- A score of 4 is useful but other questions may be more important.\n"
            "- A score of 5 is reserved for must-answer, central questions.\n"
            "- Avoid bias toward high scores. Follow these instructions carefully.\n"
            "The score should strictly be an integer from 1 to 5.\n"
            "score:"
        )

        messages = [LLMMessage(role="user", content=saliency_prompt)]
        response = llm.generate(messages=messages, temperature=0)
        score = "".join(filter(str.isdigit, response.content)).strip()
        return int(score) if score else None

    except Exception as e:
        logger.error(f"[saliency] {e}")
        return None


def get_log_probs(llm_client, prompt):
    """Query GPT-4o-mini API to get log probabilities of the next token."""
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
        )
        log_probs = response.choices[0].logprobs.content[0].top_logprobs
        return {entry.token: np.exp(entry.logprob) for entry in log_probs}
    except Exception as e:
        logger.error(f"[EIG/log_prob] {e}")
        return None


def compute_entropy(prob_dist):
    """Compute Shannon entropy for a given probability distribution."""
    prob_values = np.array(list(prob_dist.values()))
    prob_values = prob_values[prob_values > 0]  # Remove zero probabilities for log stability
    return -np.sum(prob_values * np.log(prob_values))


def compute_eig_score(llm_client, article, question, answer):
    """Compute Expected Information Gain (EIG) using GPT-4o-mini's log probabilities.

    - Prior entropy: Before seeing the answer.
    - Posterior entropy: After conditioning on the first token of the answer.
    """
    eig_prompt = (
        f"Imagine you are a reader who is reading the article. You come across a question and need to answer it.\n"
        f"article: {article}\n"
        f"question: {question}\n"
        f"answer:"
    )
    try:
        prior = get_log_probs(llm_client, eig_prompt)
        if not prior:
            return None
        prior_entropy = compute_entropy(prior)

        first_token = answer.strip().split()[0]
        conditioned_prompt = f"{eig_prompt} {first_token}"
        posterior = get_log_probs(llm_client, conditioned_prompt)
        if not posterior:
            return None
        posterior_entropy = compute_entropy(posterior)

        return prior_entropy - posterior_entropy
    except Exception as e:
        logger.error(f"[EIG] {e}")
        return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    llm_generator = OpenAIGenerator(oai_api_key=os.getenv("OPENAI_API_KEY", ""), model=args.evaluate_model_name)

    simulator = Simulator(generator=llm_generator)
    llm_client = llm_generator.sync_client  # For raw log_prob access

    qa_data = read_jsonl(args.qa_file)
    base_name = os.path.splitext(os.path.basename(args.qa_file))[0].replace("_qa_pairs", "")

    # 1. Group questions by subject/chapter/section
    subject = qa_data[0]["subject"]
    questions_by_section = {}
    for item in qa_data:
        sec = str(item.get("section", "1"))
        questions_by_section.setdefault(sec, []).append(item)

    # 2. Get textbook info (last 5 chapters only)
    filtered = [d for d in textbook_data if d["subject"] == subject]
    test_data = sorted(filtered, key=lambda x: x["chapter"])[-5:]

    # 3. Run simulator for utility
    results = []
    for chapter_data in test_data:
        exam_questions = chapter_data["llm_parsed_results"]["questions"]
        chapter = chapter_data["chapter"]
        try:
            utilities, _, _ = simulator.generate(
                eval_questions=exam_questions,
                sections={},  # You can change to full doc
                generated_questions=questions_by_section,
            )
            for qid, u in utilities.items():
                results.append(
                    {
                        "qid": qid,
                        "question": u["question"],
                        "answer": u["answer"],
                        "section": u["section"],
                        "utility": u["utility"],
                        "chapter": chapter,
                        "subject": subject,
                    }
                )
        except Exception as e:
            logger.error(f"[utility] Simulation failed: {e}")
            continue

        # 4. Compute saliency & EIG if needed
        if args.include_saliency or args.include_eig:
            for item in results:
                try:
                    section = int(item["section"])
                    chapter = item["chapter"]
                    question = item["question"]
                    answer = item["answer"]

                    chapter_data = next(d for d in textbook_data if d["subject"] == subject and d["chapter"] == chapter)
                    sections = chapter_data["llm_parsed_results"]["sections"]

                    context = "".join([sections[str(i)]["content"] for i in range(1, section)])
                    anchor = sections[str(section)]["content"]
                    article = context + anchor

                    if args.include_saliency:
                        item["saliency"] = get_saliency_score(llm_generator, article, question, answer)
                    if args.include_eig:
                        item["eig"] = compute_eig_score(llm_client, article, question, answer)

                except Exception as e:
                    logger.error(f"[saliency/EIG] failed for {item.get('qid')}: {e}")
                    continue

    out_path = os.path.join(args.output_dir, f"{base_name}_question_metrics.jsonl")
    write_jsonl(results, out_path)
    print(f"✅ Saved all metrics to: {out_path}")


if __name__ == "__main__":
    main()
