#!/usr/bin/env python3
"""
Tracking logger for QUEST training pipeline
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class TrackingLogger:
    """Centralized logging for QUEST training pipeline."""
    
    def __init__(self, tracking_file: str, chapter_log_file: Optional[str] = None,
                 iteration: int = 1, subject: Optional[str] = None, 
                 chapter_id: Optional[str] = None):
        self.tracking_file = tracking_file
        self.chapter_log_file = chapter_log_file
        self.iteration = iteration
        self.subject = subject
        self.chapter_id = chapter_id
        
        # Create tracking file if it doesn't exist
        if tracking_file and not os.path.exists(tracking_file):
            with open(tracking_file, "w") as f:
                pass
        
        # Create chapter log file if specified
        if chapter_log_file:
            os.makedirs(os.path.dirname(chapter_log_file), exist_ok=True)
            if os.path.exists(chapter_log_file):
                os.remove(chapter_log_file)
    
    def set_iteration(self, iteration: int):
        """Set the current iteration."""
        self.iteration = iteration
    
    def set_subject(self, subject: str):
        """Set the current subject."""
        self.subject = subject
    
    def set_chapter(self, chapter_id: str):
        """Set the current chapter."""
        self.chapter_id = chapter_id
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iteration += 1
    
    def log(self, entry_type: str, data: Dict[str, Any]):
        """Log an entry with standard metadata."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "iteration": self.iteration,
            **data
        }
        
        if self.subject:
            entry["subject"] = self.subject
        if self.chapter_id:
            entry["chapter_id"] = self.chapter_id
        
        # Write to main tracking file
        if self.tracking_file:
            with open(self.tracking_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        
        # Write to chapter log file
        if self.chapter_log_file:
            with open(self.chapter_log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def log_iteration_start(self, total_iterations: int, model_name: str, 
                           subjects: list, threshold: float, use_document_for_simulate: bool, 
                           num_questions_per_section: int):
        """Log iteration start."""
        self.log("iteration_start", {
            "total_iterations": total_iterations,
            "model_name": model_name,
            "subjects": subjects,
            "threshold": threshold,
            "use_document_for_simulate": use_document_for_simulate,
            "num_questions_per_section": num_questions_per_section
        })
    
    def log_subject_start(self, total_chapters: int):
        """Log subject processing start."""
        self.log("subject_start", {
            "total_chapters": total_chapters
        })
    
    def log_chapter_start(self, chapter_index: int, total_sections: int, total_exam_questions: int):
        """Log chapter processing start."""
        self.log("chapter_start", {
            "chapter_index": chapter_index,
            "total_sections": total_sections,
            "total_exam_questions": total_exam_questions
        })
    
    def log_section_questions(self, section_id: str, anchor: str, questions: list, answers: list, prompt: str):
        """Log question generation for a section."""
        self.log("section_questions_generated", {
            "section_id": section_id,
            "anchor": anchor,
            "questions": questions,
            "answers": answers,
            "prompt": prompt
        })
    
    def log_simulation_results(self, baseline_score: float, all_questions_score: float, 
                              total_generated_questions: int, exam_questions: dict,
                              utilities: dict, individual_scores: dict):
        """Log simulation results."""
        self.log("simulation_results", {
            "baseline_score": baseline_score,
            "all_questions_score": all_questions_score,
            "total_generated_questions": total_generated_questions,
            "exam_questions": exam_questions,
            "utilities": utilities,
            "baseline_individual_scores": individual_scores["baseline"],
            "all_questions_individual_scores": individual_scores["all_questions"]
        })
    
    def log_generated_question_detail(self, qid: str, question: str, answer: str, section: str,
                                     single_gain: float, all_but_one_gain: float, utility: float,
                                     exam_questions: dict):
        """Log detailed tracking for a generated question."""
        self.log("generated_question_detail", {
            "generated_question_id": qid,
            "question": question,
            "answer": answer,
            "section": section,
            "single_gain": single_gain,
            "all_but_one_gain": all_but_one_gain,
            "utility": utility,
            "exam_questions": exam_questions
        })
    
    def log_all_generated_questions_summary(self, threshold: float, all_questions: list, high_utility_count: int):
        """Log summary of all generated questions."""
        self.log("all_generated_questions_summary", {
            "threshold": threshold,
            "total_generated_questions": len(all_questions),
            "high_utility_questions": high_utility_count,
            "low_utility_questions": len(all_questions) - high_utility_count,
            "all_questions": all_questions
        })
    
    def log_simulation_error(self, error: str):
        """Log simulation error."""
        self.log("simulation_error", {
            "error": error
        })
    
    def log_chapter_complete(self, high_utility_count: int):
        """Log chapter completion."""
        self.log("chapter_complete", {
            "high_utility_questions_count": high_utility_count
        })
    
    def log_subject_complete(self, total_high_utility: int):
        """Log subject completion."""
        self.log("subject_complete", {
            "total_high_utility_questions": total_high_utility
        })
    
    def log_no_high_utility_questions(self, threshold: float):
        """Log when no high utility questions are found."""
        self.log("no_high_utility_questions", {
            "threshold": threshold
        })
    
    def log_high_utility_questions_summary(self, high_utility_questions: list):
        """Log summary of high utility questions."""
        self.log("high_utility_questions_summary", {
            "total_high_utility_questions": len(high_utility_questions),
            "questions": high_utility_questions
        })
    
    def log_training_data_created(self, training_data_file: str, training_data_count: int):
        """Log training data creation."""
        self.log("training_data_created", {
            "training_data_file": training_data_file,
            "training_data_count": training_data_count
        })
    
    def log_fine_tuning_job_start(self, job_id: str, model_name: str, training_file_id: str):
        """Log fine-tuning job start."""
        self.log("fine_tuning_job_start", {
            "job_id": job_id,
            "model_name": model_name,
            "training_file_id": training_file_id
        })
    
    def log_fine_tuning_status(self, job_id: str, status: str):
        """Log fine-tuning status update."""
        self.log("fine_tuning_status", {
            "job_id": job_id,
            "status": status
        })
    
    def log_fine_tuning_success(self, job_id: str, old_model: str, 
                               new_model: str, metadata_file: str):
        """Log fine-tuning success."""
        self.log("fine_tuning_success", {
            "job_id": job_id,
            "old_model": old_model,
            "new_model": new_model,
            "metadata_file": metadata_file
        })
    
    def log_fine_tuning_failed(self, job_id: str):
        """Log fine-tuning failure."""
        self.log("fine_tuning_failed", {
            "job_id": job_id
        })
    
    def log_fine_tuning_no_model(self, job_id: str):
        """Log when no fine-tuned model is returned."""
        self.log("fine_tuning_no_model", {
            "job_id": job_id
        })
    
    def log_iteration_complete(self, final_model: str):
        """Log iteration completion."""
        self.log("iteration_complete", {
            "final_model": final_model
        })
    
    def log_all_iterations_complete(self, total_iterations: int, final_model: str):
        """Log all iterations completion."""
        self.log("all_iterations_complete", {
            "total_iterations": total_iterations,
            "final_model": final_model
        })
    
    def log_test_mode_complete(self, total_high_utility_questions: int):
        """Log test mode completion."""
        self.log("test_mode_complete", {
            "total_high_utility_questions": total_high_utility_questions,
            "message": "Test mode completed - fine-tuning skipped"
        }) 