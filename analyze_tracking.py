#!/usr/bin/env python3
"""
Utility script to analyze the tracking outputs from run_train_quest.py
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any


def load_tracking_data(filename: str) -> List[Dict[str, Any]]:
    """Load tracking data from JSONL file."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def analyze_tracking_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tracking data and return summary statistics."""
    analysis = {
        "total_entries": len(data),
        "entry_types": defaultdict(int),
        "iterations": set(),
        "subjects": set(),
        "chapters": set(),
        "questions_generated": 0,
        "high_utility_questions": 0,
        "simulation_errors": 0,
        "fine_tuning_jobs": [],
        "baseline_scores": [],
        "utility_scores": [],
        "single_gain_scores": [],
        "all_but_one_scores": [],
        "individual_exam_scores": [],
        "generated_question_details": [],
        "all_generated_questions_summaries": []
    }
    
    for entry in data:
        entry_type = entry.get("type", "unknown")
        analysis["entry_types"][entry_type] += 1
        
        if "iteration" in entry:
            analysis["iterations"].add(entry["iteration"])
        
        if "subject" in entry:
            analysis["subjects"].add(entry["subject"])
        
        if "chapter_id" in entry:
            analysis["chapters"].add(entry["chapter_id"])
        
        if entry_type == "section_questions_generated":
            analysis["questions_generated"] += len(entry.get("questions", []))
        
        elif entry_type == "question_utility":
            analysis["utility_scores"].append(entry.get("utility", 0))
            analysis["single_gain_scores"].append(entry.get("single_gain", 0))
            analysis["all_but_one_scores"].append(entry.get("all_but_one_gain", 0))
        
        elif entry_type == "simulation_results":
            # Extract baseline scores
            analysis["baseline_scores"].append(entry.get("baseline_score", 0))
            
            # Extract individual exam question scores if available
            baseline_scores = entry.get("baseline_individual_scores", {})
            all_questions_scores = entry.get("all_questions_individual_scores", {})
            
            for qid, score_data in baseline_scores.items():
                analysis["individual_exam_scores"].append({
                    "question_id": qid,
                    "baseline_score": float(score_data.get("score", 0)),
                    "all_questions_score": float(all_questions_scores.get(qid, {}).get("score", 0)) if qid in all_questions_scores else 0,
                    "question": score_data.get("question", ""),
                    "ground_truth": score_data.get("answer", ""),
                    "baseline_prediction": score_data.get("prediction", ""),
                    "all_questions_prediction": all_questions_scores.get(qid, {}).get("prediction", "") if qid in all_questions_scores else ""
                })
        
        elif entry_type == "generated_question_detail":
            # Extract detailed generated question information
            analysis["generated_question_details"].append({
                "generated_question_id": entry.get("generated_question_id"),
                "question": entry.get("question"),
                "answer": entry.get("answer"),
                "section": entry.get("section"),
                "utility": entry.get("utility"),
                "single_gain": entry.get("single_gain"),
                "all_but_one_gain": entry.get("all_but_one_gain"),
                "exam_questions": entry.get("exam_questions", {})
            })
        
        elif entry_type == "all_generated_questions_summary":
            # Extract all generated questions summary
            analysis["all_generated_questions_summaries"].append({
                "chapter_id": entry.get("chapter_id"),
                "threshold": entry.get("threshold"),
                "total_generated_questions": entry.get("total_generated_questions"),
                "high_utility_questions": entry.get("high_utility_questions"),
                "low_utility_questions": entry.get("low_utility_questions"),
                "all_questions": entry.get("all_questions", [])
            })
        
        elif entry_type == "high_utility_questions_summary":
            analysis["high_utility_questions"] += entry.get("total_high_utility_questions", 0)
        
        elif entry_type == "simulation_error":
            analysis["simulation_errors"] += 1
        
        elif entry_type == "fine_tuning_job_start":
            analysis["fine_tuning_jobs"].append({
                "iteration": entry.get("iteration"),
                "job_id": entry.get("job_id"),
                "model_name": entry.get("model_name")
            })
    
    # Convert sets to sorted lists for JSON serialization
    analysis["iterations"] = sorted(list(analysis["iterations"]))
    analysis["subjects"] = sorted(list(analysis["subjects"]))
    analysis["chapters"] = sorted(list(analysis["chapters"]))
    
    return analysis


def print_analysis_summary(analysis: Dict[str, Any]):
    """Print a formatted summary of the analysis."""
    print("=" * 60)
    print("TRACKING DATA ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal entries: {analysis['total_entries']}")
    print(f"Entries by type:")
    for entry_type, count in sorted(analysis['entry_types'].items()):
        print(f"  {entry_type}: {count}")
    
    print(f"\nIterations processed: {len(analysis['iterations'])}")
    print(f"Subjects: {', '.join(analysis['subjects'])}")
    print(f"Chapters processed: {len(analysis['chapters'])}")
    
    print(f"\nQuestions generated: {analysis['questions_generated']}")
    print(f"High utility questions: {analysis['high_utility_questions']}")
    print(f"Simulation errors: {analysis['simulation_errors']}")
    
    if analysis['baseline_scores']:
        avg_baseline = sum(analysis['baseline_scores']) / len(analysis['baseline_scores'])
        print(f"Average baseline score: {avg_baseline:.4f}")
    
    if analysis['utility_scores']:
        avg_utility = sum(analysis['utility_scores']) / len(analysis['utility_scores'])
        max_utility = max(analysis['utility_scores'])
        min_utility = min(analysis['utility_scores'])
        print(f"Utility scores - Avg: {avg_utility:.4f}, Max: {max_utility:.4f}, Min: {min_utility:.4f}")
    
    if analysis['single_gain_scores']:
        avg_single = sum(analysis['single_gain_scores']) / len(analysis['single_gain_scores'])
        print(f"Average single gain: {avg_single:.4f}")
    
    if analysis['all_but_one_scores']:
        avg_ab1 = sum(analysis['all_but_one_scores']) / len(analysis['all_but_one_scores'])
        print(f"Average all-but-one gain: {avg_ab1:.4f}")
    
    if analysis['individual_exam_scores']:
        print(f"\nIndividual exam question scores: {len(analysis['individual_exam_scores'])}")
        baseline_improvements = []
        for score_data in analysis['individual_exam_scores']:
            improvement = score_data['all_questions_score'] - score_data['baseline_score']
            baseline_improvements.append(improvement)
        
        if baseline_improvements:
            avg_improvement = sum(baseline_improvements) / len(baseline_improvements)
            max_improvement = max(baseline_improvements)
            min_improvement = min(baseline_improvements)
            print(f"  Average improvement per exam question: {avg_improvement:.4f}")
            print(f"  Max improvement: {max_improvement:.4f}")
            print(f"  Min improvement: {min_improvement:.4f}")
    
    if analysis['generated_question_details']:
        print(f"\nGenerated question details: {len(analysis['generated_question_details'])}")
        total_exam_questions = 0
        for detail in analysis['generated_question_details']:
            total_exam_questions += len(detail['exam_questions'])
        print(f"  Total exam question evaluations: {total_exam_questions}")
    
    if analysis['all_generated_questions_summaries']:
        print(f"\nAll generated questions summaries: {len(analysis['all_generated_questions_summaries'])}")
        total_all_questions = 0
        total_high_utility = 0
        total_low_utility = 0
        for summary in analysis['all_generated_questions_summaries']:
            total_all_questions += summary['total_generated_questions']
            total_high_utility += summary['high_utility_questions']
            total_low_utility += summary['low_utility_questions']
        print(f"  Total generated questions: {total_all_questions}")
        print(f"  High utility questions: {total_high_utility}")
        print(f"  Low utility questions: {total_low_utility}")
        if total_all_questions > 0:
            success_rate = (total_high_utility / total_all_questions) * 100
            print(f"  Success rate: {success_rate:.1f}%")
    
    print(f"\nFine-tuning jobs: {len(analysis['fine_tuning_jobs'])}")
    for job in analysis['fine_tuning_jobs']:
        print(f"  Iteration {job['iteration']}: {job['job_id']} ({job['model_name']})")


def filter_by_type(data: List[Dict[str, Any]], entry_type: str) -> List[Dict[str, Any]]:
    """Filter tracking data by entry type."""
    return [entry for entry in data if entry.get("type") == entry_type]


def main():
    parser = argparse.ArgumentParser(description="Analyze tracking outputs from run_train_quest.py")
    parser.add_argument("tracking_file", help="JSONL tracking file to analyze")
    parser.add_argument("--filter-type", help="Filter by entry type (e.g., 'question_utility')")
    parser.add_argument("--output", help="Output file for filtered results (JSON)")
    
    args = parser.parse_args()
    
    # Load and analyze data
    data = load_tracking_data(args.tracking_file)
    analysis = analyze_tracking_data(data)
    
    # Print summary
    print_analysis_summary(analysis)
    
    # Filter if requested
    if args.filter_type:
        filtered_data = filter_by_type(data, args.filter_type)
        print(f"\nFiltered entries of type '{args.filter_type}': {len(filtered_data)}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(filtered_data, f, indent=2)
            print(f"Filtered data saved to: {args.output}")
        else:
            # Print first few entries as example
            for i, entry in enumerate(filtered_data[:3]):
                print(f"\nEntry {i+1}:")
                print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main() 