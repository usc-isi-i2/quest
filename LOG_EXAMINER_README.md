# Log Examiner Web App

A simple Flask web application for examining generated question details from QUEST training logs.

## Features

- **Browse All Entries**: View a list of all generated question detail entries with key information
- **Detailed View**: Examine individual entries with full context
- **Navigation**: Navigate between entries with Previous/Next buttons or keyboard arrows
- **Jump to Entry**: Directly jump to any entry by number
- **Scoring Analysis**: View how each generated question affects exam performance
- **Quality Assessment**: Review generated questions and answers for quality

## Installation

1. Install Flask:
```bash
pip install flask
```

2. Run the web app:
```bash
# Start without specifying a file (recommended)
python log_examiner.py

# Or specify a specific log file
python log_examiner.py --log-file tracking_outputs_20241201_143022.jsonl
```

## Usage

### Basic Usage
```bash
# Start the web app (will show file selection)
python log_examiner.py

# Start with a specific log file
python log_examiner.py --log-file tracking_outputs_20241201_143022.jsonl

# Use custom port
python log_examiner.py --port 8080

# Run on all interfaces (for remote access)
python log_examiner.py --host 0.0.0.0
```

### Command Line Options
- `--log-file`: Path to a specific tracking log file (optional)
- `--port`: Port to run the web app on (default: 5000)
- `--host`: Host to run the web app on (default: 127.0.0.1)
- `--debug`: Run in debug mode

## Interface

### File Selection Page
- Automatically discovers all `.jsonl` files in the current directory and `logs/` folder
- Shows file information including:
  - File size and modification date
  - File location (current directory or logs folder)
  - Quick actions to load or get file info
- Sorted by modification time (newest first)

### File Info Page
- Detailed statistics about a log file
- Shows entry counts by type
- Displays file metadata
- Quick access to load the file for examination

### Main Page (Entry List)
- Shows a list of all generated question detail entries
- Each entry displays:
  - Entry number and timestamp
  - Subject and chapter information
  - Generated question (truncated)
  - Generated answer (truncated)
  - Utility scores with color coding
- Shows current file being examined with option to change files

### Detailed Entry View
- **Question & Answer**: Full generated question and answer text
- **Utility Scores**: 
  - Utility score (color-coded: green=positive, red=negative)
  - Single gain score
  - All-but-one gain score
- **Exam Impact**: How the generated question affects each exam question
  - Baseline performance (no questions)
  - All questions performance
  - Single question performance
  - Leave-one-out performance

### Navigation
- **Previous/Next buttons**: Navigate between entries
- **Keyboard arrows**: Left/Right arrow keys for navigation
- **Jump input**: Enter any entry number to jump directly
- **Back to List**: Return to the main listing page

## What to Look For

### Question Quality
- **Clarity**: Is the question clear and well-formed?
- **Relevance**: Does it relate to the subject matter?
- **Difficulty**: Is it appropriately challenging?

### Answer Quality
- **Accuracy**: Is the answer correct?
- **Completeness**: Does it fully address the question?
- **Clarity**: Is the explanation clear and understandable?

### Scoring Fairness
- **Baseline vs All Questions**: Does adding this question improve performance?
- **Single Question Impact**: How much does this specific question help?
- **Leave-One-Out**: How much does removing this question hurt performance?
- **Consistency**: Are the scores reasonable across different scenarios?

### Utility Assessment
- **Positive Utility**: Questions that improve exam performance
- **Negative Utility**: Questions that hurt exam performance
- **Zero Utility**: Questions with minimal impact

## Example Analysis

When examining an entry, consider:

1. **Question Quality**: "What is the chemical formula for water?" → Good, clear question
2. **Answer Quality**: "H2O" → Accurate but could be more detailed
3. **Utility Score**: +0.15 → Positive impact on exam performance
4. **Exam Impact**: 
   - Baseline: 0.6 → All Questions: 0.75 (improvement)
   - Single Question: 0.72 (significant individual contribution)
   - Leave-One-Out: 0.73 (removing hurts slightly)

## Tips for Efficient Review

1. **Start with High Utility**: Focus on questions with positive utility scores
2. **Check Patterns**: Look for similar questions that consistently perform well
3. **Examine Failures**: Review negative utility questions to understand what doesn't work
4. **Compare Scenarios**: See how questions perform in different contexts
5. **Use Keyboard Navigation**: Arrow keys make navigation much faster

## Troubleshooting

- **No entries found**: Ensure the log file contains `generated_question_detail` entries
- **Web app won't start**: Check if the port is already in use
- **Slow loading**: Large log files may take time to load initially
- **Display issues**: Try refreshing the page or using a different browser 