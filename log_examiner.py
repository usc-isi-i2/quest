#!/usr/bin/env python3
"""
Log Examiner Web App
Examines generated_question_detail entries from tracking logs
"""

import json
import os
import argparse
from flask import Flask, render_template, request, redirect, url_for
from typing import List, Dict, Any, Optional
import re
import glob
from datetime import datetime

app = Flask(__name__)

class LogExaminer:
    """Examines tracking log files for generated question details."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.entries = []
        self.current_index = 0
        self.load_entries()
    
    def load_entries(self):
        """Load all generated_question_detail entries from the log file."""
        if not os.path.exists(self.log_file):
            print(f"Log file not found: {self.log_file}")
            return
        
        self.entries = []
        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if entry.get('type') == 'generated_question_detail':
                        entry['line_number'] = line_num
                        self.entries.append(entry)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(self.entries)} generated question detail entries")
    
    def get_entry(self, index: int) -> Optional[Dict[str, Any]]:
        """Get entry at specific index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None
    
    def get_total_entries(self) -> int:
        """Get total number of entries."""
        return len(self.entries)
    
    def format_exam_questions(self, exam_questions: Dict) -> str:
        """Format exam questions for display."""
        formatted = []
        for exam_qid, exam_data in exam_questions.items():
            exam_info = f"<strong>Exam Question {exam_qid}:</strong><br>"
            exam_info += f"<em>Question:</em> {exam_data.get('question', 'N/A')}<br>"
            exam_info += f"<em>Ground Truth:</em> {exam_data.get('ground_truth', 'N/A')}<br>"
            
            # Add scoring details
            for scenario in ['baseline', 'all_questions', 'single_one', 'leave_one_out']:
                if scenario in exam_data:
                    scenario_data = exam_data[scenario]
                    exam_info += f"<em>{scenario.replace('_', ' ').title()}:</em> "
                    exam_info += f"Prediction: {scenario_data.get('prediction', 'N/A')}, "
                    exam_info += f"Score: {scenario_data.get('score', 'N/A')}, "
                    exam_info += f"Gain: {scenario_data.get('gain', 'N/A')}<br>"
                else: 
                    exam_info += f"<em>{scenario.replace('_', ' ').title()}:</em> N/A<br>"
            
            formatted.append(exam_info)
        
        return "<br><br>".join(formatted)

def find_log_files():
    """Find all available log files."""
    log_files = []
    
    # Look for files in current directory
    current_dir_files = glob.glob("*.jsonl")
    for file in current_dir_files:
        if os.path.isfile(file):
            stat = os.stat(file)
            log_files.append({
                'path': file,
                'name': file,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'type': 'current'
            })
    
    # Look for files in logs/ directory
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        logs_files = glob.glob(os.path.join(logs_dir, "*.jsonl"))
        for file in logs_files:
            if os.path.isfile(file):
                stat = os.stat(file)
                log_files.append({
                    'path': file,
                    'name': os.path.basename(file),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'type': 'logs'
                })
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x['modified'], reverse=True)
    return log_files

def get_log_file_info(log_file_path: str) -> Dict[str, Any]:
    """Get information about a log file."""
    if not os.path.exists(log_file_path):
        return {'error': 'File not found'}
    
    info = {
        'path': log_file_path,
        'name': os.path.basename(log_file_path),
        'size': os.path.getsize(log_file_path),
        'modified': datetime.fromtimestamp(os.path.getmtime(log_file_path))
    }
    
    # Count entries by type
    entry_counts = {}
    total_lines = 0
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                total_lines += 1
                try:
                    entry = json.loads(line.strip())
                    entry_type = entry.get('type', 'unknown')
                    entry_counts[entry_type] = entry_counts.get(entry_type, 0) + 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        info['error'] = str(e)
        return info
    
    info['total_lines'] = total_lines
    info['entry_counts'] = entry_counts
    info['generated_question_details'] = entry_counts.get('generated_question_detail', 0)
    
    return info

# Global examiner instance
examiner = None

@app.route('/')
def index():
    """Main page with file selection or entry list."""
    global examiner
    
    # If no examiner is loaded, show file selection
    if examiner is None:
        log_files = find_log_files()
        return render_template('file_selection.html', log_files=log_files)
    
    # If examiner is loaded, show entry list
    entries = examiner.entries
    return render_template('index.html', 
                         entries=entries, 
                         total=examiner.get_total_entries(),
                         current_file=examiner.log_file)

@app.route('/change_file')
def change_file():
    """Clear current examiner and go back to file selection."""
    global examiner
    examiner = None
    return redirect(url_for('index'))

@app.route('/load/<path:log_file>')
def load_log_file(log_file):
    """Load a specific log file."""
    global examiner
    
    # Ensure the file path is safe
    if '..' in log_file or log_file.startswith('/'):
        return "Invalid file path", 400
    
    examiner = LogExaminer(log_file)
    
    if examiner.get_total_entries() == 0:
        return f"No generated_question_detail entries found in {log_file}", 404
    
    return redirect(url_for('index'))

@app.route('/file_info/<path:log_file>')
def file_info(log_file):
    """Get information about a log file."""
    # Ensure the file path is safe
    if '..' in log_file or log_file.startswith('/'):
        return "Invalid file path", 400
    
    info = get_log_file_info(log_file)
    return render_template('file_info.html', info=info)

@app.route('/view/<int:entry_index>')
def view_entry(entry_index):
    """View a specific entry."""
    global examiner
    if examiner is None:
        return "No log file loaded."
    
    entry = examiner.get_entry(entry_index)
    if entry is None:
        return "Entry not found."
    
    # Format exam questions for display
    exam_questions_html = examiner.format_exam_questions(entry.get('exam_questions', {}))
    
    return render_template('view_entry.html',
                         entry=entry,
                         entry_index=entry_index,
                         total_entries=examiner.get_total_entries(),
                         exam_questions_html=exam_questions_html)

@app.route('/navigate/<direction>')
def navigate(direction):
    """Navigate to next/previous entry."""
    current_index = request.args.get('current', 0, type=int)
    
    if direction == 'next':
        new_index = current_index + 1
    elif direction == 'prev':
        new_index = current_index - 1
    else:
        new_index = current_index
    
    # Ensure index is within bounds
    if examiner:
        new_index = max(0, min(new_index, examiner.get_total_entries() - 1))
    
    return redirect(url_for('view_entry', entry_index=new_index))

@app.route('/jump/<int:target_index>')
def jump_to_entry(target_index):
    """Jump to a specific entry index."""
    if examiner:
        target_index = max(0, min(target_index, examiner.get_total_entries() - 1))
    return redirect(url_for('view_entry', entry_index=target_index))

@app.template_filter('format_score')
def format_score(score, is_gain=False):
    """Format score for display."""
    if score is None:
        return "N/A"
    try:
        # if is_gain and >0, show dark green, if <0, show dark red, otherwise show black
        if is_gain and float(score) > 0:
            return f'<span style="color: darkgreen">{float(score):.3f}</span>'
        elif is_gain and float(score) < 0:
            return f'<span style="color: darkred">{float(score):.3f}</span>'
        else:
            return f"{float(score):.3f}"
    except (ValueError, TypeError):
        return str(score)

@app.template_filter('format_utility')
def format_utility(utility):
    """Format utility score for display."""
    if utility is None:
        return "N/A"
    try:
        utility_float = float(utility)
        color = "green" if utility_float > 0 else "red" if utility_float < 0 else "black"
        return f'<span style="color: {color}">{utility_float:.3f}</span>'
    except (ValueError, TypeError):
        return str(utility)

@app.template_filter('format_utility_safe')
def format_utility_safe(utility):
    """Format utility score for display without HTML."""
    if utility is None:
        return "N/A"
    try:
        utility_float = float(utility)
        return f"{utility_float:.3f}"
    except (ValueError, TypeError):
        return str(utility)

@app.template_filter('utility_color_class')
def utility_color_class(utility):
    """Get CSS class for utility score color."""
    if utility is None:
        return "utility-neutral"
    try:
        utility_float = float(utility)
        if utility_float > 0:
            return "utility-positive"
        elif utility_float < 0:
            return "utility-negative"
        else:
            return "utility-neutral"
    except (ValueError, TypeError):
        return "utility-neutral"

@app.template_filter('gain_color_class')
def gain_color_class(gain):
    """Get CSS class for gain score color."""
    if gain is None:
        return "gain-neutral"
    try:
        gain_float = float(gain)
        if gain_float > 0:
            return "gain-positive"
        elif gain_float < 0:
            return "gain-negative"
        else:
            return "gain-neutral"
    except (ValueError, TypeError):
        return "gain-neutral"

def main():
    parser = argparse.ArgumentParser(description='Examine tracking log files')
    parser.add_argument('--log-file', help='Path to a specific tracking log file (optional)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on')
    parser.add_argument('--host', default='127.0.0.1', help='Host to run the web app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    global examiner
    
    # If a specific log file is provided, load it
    if args.log_file:
        examiner = LogExaminer(args.log_file)
        if examiner.get_total_entries() == 0:
            print(f"No generated_question_detail entries found in {args.log_file}")
            return
        print(f"Loaded {examiner.get_total_entries()} generated question detail entries from {args.log_file}")
    
    print(f"Starting web app on http://{args.host}:{args.port}")
    print("Navigate to the URL above to examine the logs")
    print("If no log file was specified, you can select one from the web interface")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 