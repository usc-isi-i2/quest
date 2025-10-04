from pathlib import Path
import json 
from utils_file import read_jsonl
import numpy as np

files = Path(".").glob("tracking*.jsonl")

for file in files:
    data = read_jsonl(file)
    
    data = [d for d in data if d["type"] == "generated_question_detail"]
    single_gain = [round(d["single_gain"], 5) for d in data]
    all_but_one_gain = [round(d["all_but_one_gain"], 5) for d in data]
    blended_utility = [round(d["utility"], 5) for d in data]
    diff = [abs(round(d["single_gain"] - d["all_but_one_gain"], 5)) for d in data]
    
    print(f"{file.name}:")
    print(f"Single gains: {single_gain} - mean: {np.mean(single_gain):.5f}")
    print(f"All-but-one gain: {all_but_one_gain} - mean: {np.mean(all_but_one_gain):.5f}")
    print(f"Blended utility: {blended_utility} - mean: {np.mean(blended_utility):.5f}")
    print(f"Diff: {diff} - mean: {np.mean(diff):.5f}")
    breakpoint() 