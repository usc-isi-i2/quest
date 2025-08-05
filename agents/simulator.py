from typing import Any, Union

import numpy as np
from collections import defaultdict

from agents.base import BaseAgent
from agents.simulator_evaluator import Evaluator
from agents.simulator_learner import Learner

from loguru import logger

class Simulator(BaseAgent[tuple[dict[str, dict[str, Union[str, Any]]], float, float, dict[str, dict[str, dict[str, dict[str, str]]]]]]):
    def __init__(self, generator, alpha=0.5):
        """Simulator agent that evaluates the utility of generated questions.

        :param generator: The question generator (as in your existing architecture).
        :param alpha: Blend weight for single-question utility vs. all-but-one utility.
        """
        super().__init__(generator)
        self.learner = Learner(generator)
        self.evaluator = Evaluator(generator)
        self.alpha = alpha  # 0 <= alpha <= 1

    def generate(
        self,
        eval_questions: dict[str, dict],
        sections: dict[str, dict],
        generated_questions: dict[str, list],
        test: bool = False,
    ) -> tuple[dict[str, dict[str, Union[str, Any]]], float, float, dict[str, dict[str, dict[str, dict[str, str]]]]]:
        r"""Computes a blended utility measure for each generated question.

          Utility(q_i) = alpha * (score({q_i}) - score({})) + (1 - alpha) * (score(all) - score(all \ {q_i}))

        :param eval_questions: The test or exam questions used to evaluate final performance.
        :param sections: Section metadata for the evaluator/learner.
        :param generated_questions: A dict of section_id -> list of question dicts (the generator output).
        :param test: If True, only get score(all).
        :return: A tuple of (utilities_dict, all_score, baseline_score, individual_scores_dict)
        """
        
        # 1) Evaluate the baseline (no additional questions)
        # TODO: Baseline score is too fragile; consider a more robust method
        baseline_predictions = self.learner.generate(eval_questions, sections, {})
        baseline_results = self.evaluator.generate(baseline_predictions, sections)
        baseline_score = np.mean([float(v["score"]) for v in baseline_results.values()])
        logger.info(f"[Baseline] No questions score: {baseline_score:.4f}")

        # 2) Flatten the generated_questions into a list for indexing
        flattened_questions = []
        for section_id, questions in generated_questions.items():
            for idx, qdict in enumerate(questions, start=1):
                qid = f"{section_id}_Q{idx}"
                flattened_questions.append((qid, section_id, qdict))

        # Make a quick lookup
        question_ids = [x[0] for x in flattened_questions]
        logger.info(f"Total questions generated: {len(question_ids)}")



        # ---------------------------------------------------------------------
        # 3) Evaluate ALL questions as a single set
        # ---------------------------------------------------------------------
        all_subset = defaultdict(list)
        for _, section_id, qdict in flattened_questions:
            all_subset[section_id].append({"question": qdict["question"], "answer": qdict["answer"]})

        all_predictions = self.learner.generate(eval_questions, sections, all_subset)
        all_results = self.evaluator.generate(all_predictions, sections)
        all_score = np.mean([float(v["score"]) for v in all_results.values()])
        logger.info(f"[All-Set] Score with all questions: {all_score:.4f}")

        if test:
            return {}, float(all_score), float(baseline_score), {"baseline": baseline_results, "all_questions": all_results, "single_questions": {}, "leave_one_out": {}}
        # if all_score <= baseline_score:
        #     logger.info("No questions improved the baseline score. Exiting.")
        #     return {}, float(all_score), float(baseline_score), {"baseline": baseline_results, "all_questions": all_results, "single_questions": {}, "leave_one_out": {}}

 

        individual_scores = {
            "baseline": baseline_results,
            "all_questions": all_results,
            "single_questions": {},
            "leave_one_out": {}
        }
        

        # ---------------------------------------------------------------------
        # 4) Single-Question Gains
        # ---------------------------------------------------------------------
        single_question_gains = {}
        for qid, section_id, qdict in flattened_questions:
            single_subset = {section_id: [{"question": qdict["question"], "answer": qdict["answer"]}]}

            single_predictions = self.learner.generate(eval_questions, sections, single_subset)
            single_results = self.evaluator.generate(single_predictions, sections)
            single_score = np.mean([float(v["score"]) for v in single_results.values()])

            gain = single_score - baseline_score
            single_question_gains[qid] = gain
            logger.info(f"[Single] {qid}: single_score={single_score:.4f}, gain={gain:.4f}")

            if qid not in individual_scores["single_questions"]:
                individual_scores["single_questions"][qid] = defaultdict(dict)

            for eval_qid, eval_qdict in eval_questions.items():
                individual_scores["single_questions"][qid][eval_qid] = {
                    "question": eval_qdict["question"],
                    "answer": eval_qdict["answer"],
                    "score": float(single_results[eval_qid]["score"]),
                    "prediction": single_results[eval_qid]["prediction"],
                    "feedback": single_results[eval_qid]["feedback"],
                    "gain": float(single_results[eval_qid]["score"]) - float(baseline_results[eval_qid]["score"])
                }

 

        # ---------------------------------------------------------------------
        # 5) All-But-One Gains
        # ---------------------------------------------------------------------
        all_minus_one_gains = {}
        # We already have score(all) = all_score
        # Now for each question q_i, evaluate all \ {q_i}
        for qid, _, _ in flattened_questions:
            subset_minus_one = {}
            for other_qid, other_section_id, other_qdict in flattened_questions:
                if other_qid == qid:
                    continue  # skip the question we're removing
                if other_section_id not in subset_minus_one:
                    subset_minus_one[other_section_id] = []
                subset_minus_one[other_section_id].append(
                    {"question": other_qdict["question"], "answer": other_qdict["answer"]}
                )

            minus_one_predictions = self.learner.generate(eval_questions, sections, subset_minus_one)
            minus_one_results = self.evaluator.generate(minus_one_predictions, sections)
            minus_one_score = np.mean([float(v["score"]) for v in minus_one_results.values()])

            # The "contribution" of q_i in the full set is:
            # all_score - minus_one_score
            contribution = all_score - minus_one_score
            all_minus_one_gains[qid] = contribution
            logger.info(f"[All-but-One] {qid}: minus_one_score={minus_one_score:.4f}, contribution={contribution:.4f}")
            
            if qid not in individual_scores["leave_one_out"]:
                individual_scores["leave_one_out"][qid] = defaultdict(dict)

            for eval_qid, eval_qdict in eval_questions.items():
                individual_scores["leave_one_out"][qid][eval_qid] = {
                    "question": eval_qdict["question"],
                    "answer": eval_qdict["answer"],
                    "score": float(minus_one_results[eval_qid]["score"]),
                    "prediction": minus_one_results[eval_qid]["prediction"],
                    "feedback": minus_one_results[eval_qid]["feedback"],
                    "gain": float(all_results[eval_qid]["score"]) - float(minus_one_results[eval_qid]["score"])
                }

 

        # ---------------------------------------------------------------------
        # 6) Combine into a single blended utility
        # ---------------------------------------------------------------------
        utilities = {}
        for qid, section_id, qdict in flattened_questions:
            s_gain = single_question_gains[qid]
            ab1_gain = all_minus_one_gains[qid]

            # blended_utility = alpha * single + (1-alpha) * all_but_one
            blended_utility = self.alpha * s_gain + (1 - self.alpha) * ab1_gain

            utilities[qid] = {
                "question": qdict["question"],
                "answer": qdict["answer"],
                "section": section_id,
                "single_gain": s_gain,
                "all_but_one_gain": ab1_gain,
                "utility": blended_utility,
                "individual_single_gains": [individual_scores["single_questions"][qid][eval_qid]["gain"] for eval_qid in eval_questions.keys()],
                "individual_leave_one_out_gains": [individual_scores["leave_one_out"][qid][eval_qid]["gain"] for eval_qid in eval_questions.keys()],
                "individual_blended_utilities": [(individual_scores["single_questions"][qid][eval_qid]["gain"] + individual_scores["leave_one_out"][qid][eval_qid]["gain"]) / 2 for eval_qid in eval_questions.keys()]
            }
            
 
            

        # ---------------------------------------------------------------------
        # 7) Print final results
        # ---------------------------------------------------------------------
        logger.info("\n--- Blended Utility Results ---")
        for qid, info in utilities.items():
            logger.info(f"{qid}:")
            logger.info(f"  Single-question gain: {info['single_gain']:.4f}")
            logger.info(f"  All-but-one gain:     {info['all_but_one_gain']:.4f}")
            logger.info(f"  Blended utility:      {info['utility']:.4f}")
            logger.info(f"  Q: {info['question']}")
            logger.info(f"  A: {info['answer']}\n")

        test_score = all_score
                
        return utilities, float(test_score), float(baseline_score), individual_scores
