import argparse
import os
import traceback

from loguru import logger
import numpy as np

from agents import Simulator
from utils_file import read_jsonl, write_jsonl
from utils_llm import LLMMessage, OpenAIGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate question quality metrics.")
    parser.add_argument("--qa_file", type=str, required=True, help="Path to qa_pairs.jsonl file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="LLM model to use.")
    parser.add_argument("--output_dir", type=str, default="q_metrics", help="Directory to save results.")
    parser.add_argument("--include_saliency", action="store_true", help="Whether to compute saliency score.")
    parser.add_argument("--include_eig", action="store_true", help="Whether to compute expected information gain.")
    parser.add_argument("--qsalience_model_name",type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="model name for qsalience")
    parser.add_argument("--qsalience_qlora_model_name", type=str, default="lingchensanwen/mistral-ins-generation-best-balanced", help="qlora model name for qsalience")
    return parser.parse_args()


def group_by_key(data, key_func):
    """Group data by a key function."""
    grouped = {}
    for item in data:
        key = key_func(item)
        grouped.setdefault(key, []).append(item)
    return grouped


def get_saliency_score(llm, article, question, answer):
    """Get saliency score using zero-shot LLM approach."""
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
    """Compute Expected Information Gain (EIG) using GPT-4o-mini's log probabilities."""
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


def build_article_context(chapter_data, section):
    """Build article context for a given section."""
    sections = chapter_data["llm_parsed_results"]["sections"]
    context = "".join([sections[str(i)]["content"] for i in range(1, section)])
    anchor = sections[str(section)]["content"]
    return context + anchor


def compute_saliency_with_fallback(qsalience, llm_generator, article, question, answer, qid):
    """Compute saliency score with fallback to zero-shot approach."""
    try:
        salience = qsalience.predict_salience(article, question)
        attempts = 1
        while salience == -1 and attempts < 3:
            salience = qsalience.predict_salience(article, question)
            if salience == -1:
                logger.error(f"Salience score is -1 for {qid}, retrying...")
                attempts += 1
            else:
                break
        if attempts == 3:
            logger.error(f"Salience score is -1 for {qid}, falling back to zero-shot approach")
            salience = get_saliency_score(llm_generator, article, question, answer)
    except Exception as e:
        logger.error(f"QSalience prediction failed for {qid}: {e}. Falling back to zero-shot approach")
        salience = get_saliency_score(llm_generator, article, question, answer)
    return salience


def process_metrics_for_item(item, textbook_data, subject, qsalience, llm_generator, llm_client, args):
    """Process saliency and EIG metrics for a single item."""
    try:
        section = int(item["section"])
        chapter = item["chapter"]
        question = item["question"]
        answer = item["answer"]

        chapter_data = next(d for d in textbook_data if d["subject"] == subject and d["chapter"] == chapter)
        article = build_article_context(chapter_data, section)

        if args.include_saliency:
            item["saliency"] = compute_saliency_with_fallback(
                qsalience, llm_generator, article, question, answer, item.get('qid')
            )
        if args.include_eig:
            item["eig"] = compute_eig_score(llm_client, article, question, answer)

    except Exception as e:
        logger.error(f"[metrics] failed for {item.get('qid')}: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")


def main():    
    args = parse_args()    
    os.makedirs(args.output_dir, exist_ok=True)
    # Load data files
    logger.info("Loading textbook data...")
    textbook_data = read_jsonl("data/data.jsonl")
    
    logger.info("Loading QA data...")
    qa_data = read_jsonl(args.qa_file)
    
    base_name = os.path.splitext(os.path.basename(args.qa_file))[0].replace("_qa_pairs", "")
    subject = qa_data[0]["subject"]

    logger.info("Initializing LLM components...")
    llm_generator = OpenAIGenerator(oai_api_key=os.getenv("OPENAI_API_KEY", ""), model=args.model_name)
    simulator = Simulator(generator=llm_generator)
    llm_client = llm_generator.sync_client

    # Check for existing results
    out_path = os.path.join(args.output_dir, f"{base_name}_question_metrics.jsonl")
    existing_results = []
    processed_items = set()  # Use (qid, chapter) tuples
    items_needing_metrics = []
    
    if os.path.exists(out_path):
        existing_results = read_jsonl(out_path)
        
        # Check which items need additional metrics computed
        for item in existing_results:
            needs_processing = False
            
            # Check if missing requested metrics
            if args.include_saliency and "saliency" not in item:
                needs_processing = True
            if args.include_eig and "eig" not in item:
                needs_processing = True
            
            if needs_processing:
                items_needing_metrics.append(item)
            else:
                if item.get("qid") and item.get("chapter"):
                    processed_items.add((item["qid"], item["chapter"]))
        
        logger.info(f"Found {len(existing_results)} existing results")
        logger.info(f"Found {len(processed_items)} complete results, {len(items_needing_metrics)} items need additional metrics")

    # Group questions by chapter and section
    questions_by_chapter = group_by_key(qa_data, lambda x: str(x['chapter']))

    # Get textbook info (last 5 chapters only)
    filtered = [d for d in textbook_data if d["subject"] == subject]
    test_data = sorted(filtered, key=lambda x: x["chapter"])[-5:]

    # Load QSalience model if needed
    qsalience = None
    if args.include_saliency:
        logger.info("Loading QSalience model (this may take 30-60 seconds)...")
        from utils_qsalience import QSalience
        qsalience = QSalience(args.qsalience_model_name, args.qsalience_qlora_model_name)
        logger.info("✅ QSalience model loaded successfully")

    # Run simulator for utility
    new_results = []
    skipped_count = 0
    
    for chapter_data in test_data:
        exam_questions = chapter_data["llm_parsed_results"]["questions"]
        chapter = chapter_data["chapter"]
        chapter_generated_questions = questions_by_chapter.get(str(chapter), [])
        questions_by_section = group_by_key(chapter_generated_questions, lambda x: str(x['section']))
        
        # Filter questions to only include unprocessed ones
        filtered_questions_by_section = {}
        for section, questions in questions_by_section.items():
            unprocessed_questions = []
            for idx, question in enumerate(questions):
                qid = f"{section}_Q{idx+1}"
                if (qid, chapter) not in processed_items:
                    unprocessed_questions.append(question)
            if unprocessed_questions:
                filtered_questions_by_section[section] = unprocessed_questions
        
        if not filtered_questions_by_section:
            logger.info(f"No unprocessed questions for chapter {chapter}, skipping")
            continue
                    
        try:
            utilities, _, _ = simulator.generate(
                eval_questions=exam_questions,
                sections={},
                generated_questions=filtered_questions_by_section,
            )
            for qid, u in utilities.items():
                new_results.append({
                    "qid": qid,
                    "question": u["question"],
                    "answer": u["answer"],
                    "section": u["section"],
                    "utility": u["utility"],
                    "chapter": chapter,
                    "subject": subject,
                })
        except Exception as e:
            logger.error(f"[utility] Simulation failed: {e}")
            continue

    # Compute additional metrics for new results
    if args.include_eig or args.include_saliency:
        for item in new_results:
            process_metrics_for_item(
                item, textbook_data, subject, qsalience, llm_generator, llm_client, args
            )
    
    # Add missing metrics to existing items that need them
    if items_needing_metrics:
        logger.info(f"Adding missing metrics to {len(items_needing_metrics)} existing items")
        for item in items_needing_metrics:
            process_metrics_for_item(
                item, textbook_data, subject, qsalience, llm_generator, llm_client, args
            )

    # Combine existing and new results
    all_results = existing_results + new_results
    
    write_jsonl(all_results, out_path)
    logger.info(f"✅ Saved {len(new_results)} new results to: {out_path}")
    if skipped_count > 0:
        logger.info(f"⏭️  Skipped {skipped_count} already processed items")


if __name__ == "__main__":
    main()
