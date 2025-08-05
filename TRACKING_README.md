# QUEST Training Tracking

This document describes the comprehensive tracking functionality added to `run_train_quest.py` that logs all intermediate outputs to a JSONL file.

## Overview

The tracking system captures detailed information about:
- Questions generated for each chapter and section
- Answers generated for those questions
- Base scores when no questions are used
- Utility scores for both single gain and leave-one-out setups
- Fine-tuning job progress and results

## Usage

### Running with Tracking

```bash
# Use default tracking file (auto-generated timestamp)
python run_train_quest.py --subject chemistry --iterations 2

# Specify custom tracking file
python run_train_quest.py --subject chemistry --iterations 2 --tracking_file my_tracking.jsonl

# Test mode - run on small subset to verify logging
python run_train_quest.py --subject chemistry --test_mode --tracking_file test_output.jsonl

# Run automated test to verify tracking functionality
python test_tracking.py
```

### Analyzing Tracking Data

Use the provided analysis script:

```bash
# Basic analysis
python analyze_tracking.py tracking_outputs_20241201_143022.jsonl

# Filter by entry type
python analyze_tracking.py tracking_outputs_20241201_143022.jsonl --filter-type question_utility

# Save filtered results
python analyze_tracking.py tracking_outputs_20241201_143022.jsonl --filter-type question_utility --output utility_questions.json
```

## Tracking Entry Types

The tracking system logs the following types of entries:

### 1. `iteration_start`
- **When**: At the beginning of each iteration
- **Data**: Model name, subjects, threshold, configuration parameters

### 2. `subject_start`
- **When**: When processing a new subject
- **Data**: Subject name, total chapters to process

### 3. `chapter_start`
- **When**: When processing a new chapter
- **Data**: Chapter ID, section count, exam question count

### 4. `section_questions_generated`
- **When**: After generating questions for a section
- **Data**: Generated questions, answers, prompts, anchor text

### 5. `simulation_results`
- **When**: After running the simulator on a chapter
- **Data**: Baseline score, all-questions score, total generated questions, utilities, individual exam question scores

### 6. `generated_question_detail`
- **When**: For each generated question after simulation
- **Data**: Detailed tracking of how each generated question affects each exam question across all scenarios (baseline, all questions, single one, leave one out)

### 7. `all_generated_questions_summary`
- **When**: After processing all generated questions for a chapter
- **Data**: Summary of all generated questions including those that don't meet the threshold, with utility scores and threshold status

### 8. `chapter_complete`
- **When**: After completing a chapter
- **Data**: High utility questions count for the chapter

### 9. `subject_complete`
- **When**: After completing a subject
- **Data**: Total high utility questions for the subject

### 10. `high_utility_questions_summary`
- **When**: After collecting all high utility questions
- **Data**: Complete list of high utility questions

### 11. `training_data_created`
- **When**: After creating training data file
- **Data**: File path, training data count

### 12. `fine_tuning_job_start`
- **When**: When starting a fine-tuning job
- **Data**: Job ID, model name, training file ID

### 13. `fine_tuning_status`
- **When**: During fine-tuning status updates
- **Data**: Job ID, current status

### 14. `fine_tuning_success` / `fine_tuning_failed`
- **When**: When fine-tuning completes
- **Data**: Job results, new model name

### 15. `iteration_complete`
- **When**: After completing an iteration
- **Data**: Final model name for the iteration

### 16. `all_iterations_complete`
- **When**: When all iterations finish
- **Data**: Final model name

### 17. `test_mode_complete`
- **When**: When test mode finishes (skips fine-tuning)
- **Data**: Total high utility questions found, completion message

## Data Structure

Each tracking entry contains:
```json
{
  "timestamp": "2024-12-01T14:30:22.123456",
  "type": "question_utility",
  "iteration": 1,
  "subject": "chemistry",
  "chapter_id": "chapter_1",
  "question_id": "1_Q1",
  "question": "What is the chemical formula for water?",
  "answer": "H2O",
  "section": "1",
  "single_gain": 0.15,
  "all_but_one_gain": 0.08,
  "utility": 0.115,
  "baseline_score": 0.65,
  "all_questions_score": 0.80
}
```



Example of generated question detail entry:
```json
{
  "timestamp": "2024-12-01T14:30:22.123456",
  "type": "generated_question_detail",
  "iteration": 1,
  "subject": "chemistry",
  "chapter_id": "chapter_1",
  "generated_question_id": "1_Q1",
  "question": "What is the chemical formula for water?",
  "answer": "H2O",
  "section": "1",
  "single_gain": 0.15,
  "all_but_one_gain": 0.08,
  "utility": 0.115,
  "exam_questions": {
    "exam_q1": {
      "question": "What is the atomic number of carbon?",
      "ground_truth": "6",
      "baseline": {
        "prediction": "6",
        "score": "1.0"
      },
      "all_questions": {
        "prediction": "6",
        "score": "1.0"
      },
      "single_one": {
        "prediction": "6",
        "score": "1.0"
      },
      "leave_one_out": {
        "prediction": "6",
        "score": "1.0"
      }
    }
  }
}
```

Example of all generated questions summary entry:
```json
{
  "timestamp": "2024-12-01T14:30:22.123456",
  "type": "all_generated_questions_summary",
  "iteration": 1,
  "subject": "chemistry",
  "chapter_id": "chapter_1",
  "threshold": 0.1,
  "total_generated_questions": 3,
  "high_utility_questions": 1,
  "low_utility_questions": 2,
  "all_questions": [
    {
      "qid": "1_Q1",
      "question": "What is the chemical formula for water?",
      "answer": "H2O",
      "section": "1",
      "utility": 0.115,
      "meets_threshold": true
    },
    {
      "qid": "1_Q2",
      "question": "What is the atomic number of carbon?",
      "answer": "6",
      "section": "1",
      "utility": 0.05,
      "meets_threshold": false
    }
  ]
}
```

## Analysis Examples

### Find High Utility Questions
```bash
python analyze_tracking.py tracking.jsonl --filter-type question_utility --output high_utility.json
```

### Get Baseline Scores
```bash
python analyze_tracking.py tracking.jsonl --filter-type simulation_results
```

### Get Simulation Results (includes individual exam scores)
```bash
python analyze_tracking.py tracking.jsonl --filter-type simulation_results --output simulation_results.json
```

### Get Generated Question Details
```bash
python analyze_tracking.py tracking.jsonl --filter-type generated_question_detail --output question_details.json
```

### Track Fine-tuning Progress
```bash
python analyze_tracking.py tracking.jsonl --filter-type fine_tuning_status
```

## Test Mode

The tracking system includes a test mode for verifying logging functionality:

### Test Mode Features:
- **Small Subset**: Processes only 1 subject, 1 chapter, 1 section
- **Quick Execution**: Completes in ~1-2 minutes
- **No Fine-tuning**: Skips the expensive fine-tuning step
- **Complete Logging**: Still logs all expected entry types
- **Verification**: Automated test script validates functionality

### Test Commands:
```bash
# Manual test mode
python run_train_quest.py --subject chemistry --test_mode --tracking_file test_output.jsonl

# Automated test with verification
python test_tracking.py
```

## Benefits

1. **Complete Audit Trail**: Every step of the process is logged with timestamps
2. **Debugging**: Easy to identify where issues occur in the pipeline
3. **Analysis**: Rich data for understanding question quality and utility patterns
4. **Reproducibility**: Full context for reproducing results
5. **Monitoring**: Real-time tracking of fine-tuning job progress
6. **Testing**: Built-in test mode for verifying logging functionality

## File Management

- Tracking files are automatically created with timestamps if not specified
- Files are appended to, so you can resume interrupted runs
- Analysis script can handle large tracking files efficiently
- Consider archiving old tracking files to save space

## Integration

The tracking system integrates seamlessly with existing functionality:
- No changes to core algorithm logic
- Minimal performance impact
- Optional feature (can be disabled by not specifying tracking file)
- Compatible with all existing command line arguments 