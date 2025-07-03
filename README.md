# Which Questions Should I Ask? Utility Estimation of Questions with LLM-based Simulations
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)  
[![arXiv](https://img.shields.io/badge/arXiv-2502.17383-b31b1b.svg)](https://arxiv.org/abs/2502.17383)

This repository contains the dataset, models, and evaluation code from our paper:  
**[Which Questions Should I Ask? Utility Estimation of Questions with LLM-based Simulations](https://arxiv.org/abs/2502.17383)**

**Authors**: [Dong-Ho Lee](https://dongholee.com/), [Hyundong Cho](https://justin-cho.com/), [Jonathan May](https://jonmay.github.io/webpage/), [Jay Pujara](https://www.jaypujara.org/)

---

## 📚 Overview

Asking effective questions is essential for learning and comprehension. 
However, evaluating and generating useful questions is challenging due to the huge space of possible questions and the lack of direct measures of their impact.

Prior work relies on indirect proxies of question quality, which do not directly assess how much a question helps a learner.  
We introduce **QUEST** (**Q**uestion **U**tility **E**stimation with **S**imulated **T**ests), a simulation framework that quantifies the *utility* of a question, its contribution to learning outcomes, by modeling how it affects a simulated learner's understanding.

**QUEST** identifies high-utility questions and uses them to fine-tune question generation models via rejection sampling.

Across five textbook domains, we find that **QUEST**-trained models produce questions that lead to **20%+ higher exam scores** compared to:
- prompt-based models grounded in instructional design literature, and
- models fine-tuned on indirect quality signals

---

## 📁 Table of Contents

1. [Setup](#setup)
2. [Data](#data)
3. [Inference](#inference) — Generates questions and evaluates their overall utility
4. [Training](#training) — Supervised fine-tuning (SFT) and QUEST training
5. [Evaluation](#evaluation) — Assesses the quality of each individual question, generated from the inference step.

   * [Utility](#utility)
   * [Saliency](#saliency)
   * [Expected Information Gain (EIG)](#expected-information-gain-eig)

---

## ⚙️ Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

---

## 📦 Data

### `data/data.jsonl`

| Field                  | Description                                             |
|------------------------|---------------------------------------------------------|
| `subject`              | Subject name (e.g., `"chemistry"`)                     |
| `chapter`              | Chapter ID (e.g., `"m50984"`)                           |
| `llm_parsed_results.sections` | Dictionary of section number → paragraph content     |
| `llm_parsed_results.questions` | Dict of question info with fields below:            |
| └─ `question`          | Question text                                           |
| └─ `answer`            | (Optional) Reference answer                             |
| └─ `relevant_sections` | List of related section numbers (LLM-inferred)          |


---

## 🧪 Inference

To generate questions and evaluate their impact on learning outcomes, run:

```bash
python run_inference.py \
  --subject chemistry \
  --qg_model_name gpt-4o-mini \
  --evaluate_model_name gpt-4o-mini \
  --mode fewshot
```

You can specify different prompting strategies and models:

| Mode      | Model                 | Description                          |
| --------- | --------------------- | ------------------------------------ |
| `default` | `gpt-4o-mini`         | **Zero-shot** baseline               |
| `cot`     | `gpt-4o-mini`         | **Chain-of-thought** prompting       |
| `fewshot` | `gpt-4o-mini`         | **Few-shot** prompting with examples |
| `default` | `sft-trained-model`   | **SFT** (Supervised Fine-Tuning)     |
| `default` | `quest-trained-model` | **QUEST** (Utility-trained model)    |

### ⚙️ Options

* `--subject`: required; one of `chemistry`, `biology`, `physics`, `economics`, etc.
* `--qg_model_name`: model used to generate questions
* `--evaluate_model_name`: model used to simulate learning
* `--mode`: prompting strategy (`default`, `cot`, `fewshot`)
* `--num_questions_per_section`: number of questions to generate per section (default: 1)
* `--use_document_for_simulate`: if set, uses full document context during simulation

---

### 📤 Output

For subject `chemistry`, output files are saved as:

* `output/{model_name}/chemistry_{mode}_performance_results.jsonl`: simulated utility scores by chapter
* `output/{model_name}/chemistry_{mode}_qa_pairs.jsonl`: generated question–answer pairs

---

### 🔧 Inference with fine-tuned SFT / QUEST model

After training (see [Training](#training)), use your fine-tuned model like this:

```python
# Load fine-tuned model name
import json
metadata = json.load(open("metadata/train_metadata_all_sft.json"))
model_name = metadata["fine_tuned_model"]
print(model_name)  # e.g., ft:gpt-4o-mini:your-team:custom-id
```

Then run:

```bash
python run_inference.py \
  --subject chemistry \
  --qg_model_name <fine_tuned_model_name> \
  --evaluate_model_name gpt-4o-mini \
  --mode default
```

---

## 🏋️ Training

We provide two training modes:

### 🧪 1. **Supervised Fine-Tuning (SFT)**

We fine-tune a base LLM (e.g., `gpt-4o-mini`) using human-authored exam questions from textbook sections.
The prompt asks the model to generate a question that helps a student understand a specific sentence in a passage.

**To prepare and run SFT training:**

```bash
python run_train_sft.py --subject all --model_name gpt-4o-mini-2024-07-18
```

* You can replace `all` with the specific subject like `--subject chemistry`.
  * `all` is for `cross-subject` setting.
* The script will create training data from all but the last 5 chapters of each subject.
* Metadata and fine-tuned model information is stored under `metadata/train_metadata_*_sft.json`.

**Training prompt format:**

```
article: <full previous context>
Student is currently reading the sentence: <anchor sentence>.
Generate a question that helps the student understand the sentence better.
Output in following JSON format:
{
    "question": question
}
```

Each fine-tuning example is a chat message pair:

* `user`: the prompt above
* `assistant`: the expected `{"question": ...}` response

Training data is saved as:

```
metadata/train_data_{subjects}_sft.jsonl
```

### 🧪 2. **QUEST Training**

**QUEST** identifies *high-utility* questions using a simulated learner and fine-tunes the model only on these high-utility examples using **rejection sampling**.

Unlike SFT, which uses all questions, QUEST selectively fine-tunes using only questions that improve simulated comprehension scores.

---

### 🚀 To run QUEST training:

```bash
python run_train_quest.py --subject all --model_name gpt-4o-mini-2024-07-18 --iterations 1 --threshold 0.1
```

* You can replace `all` with the specific subject (e.g., `--subject chemistry`)
* `--iterations`: how many fine-tuning + filtering loops to run (default: 1)
* `--threshold`: minimum utility score (between 0 and 1) for selecting questions
---

### 🔍 What does it do?

1. **Generate questions per section** from textbook chapters using the current model.
2. **Evaluate the utility** of each question using a simulated learner.
3. **Filter out low-utility questions** (utility ≤ threshold).
4. **Fine-tune the model** on the remaining high-utility examples.
5. **Repeat** (if multiple iterations).

---

### 📁 Outputs

* **Training data per iteration**:

  ```
  metadata/train_data_{subject}_iter_{i}_thresh_{threshold}.jsonl
  ```

* **Trained model metadata**:

  ```
  metadata/train_metadata_{subject}_iter_{i}_thresh_{threshold}.json
  ```

Each metadata file contains:

```json
{
  "iteration": 1,
  "subjects": ["chemistry"],
  "threshold": 0.1,
  "base_model_used": "gpt-4o-mini-2024-07-18",
  "fine_tuned_model": "ft:gpt-4o-mini:your-team:custom-id"
}
```

---

## 🧪 Evaluation

After generating questions with `run_inference.py`, you can evaluate the **individual quality** of each question using `run_eval.py`.
This helps analyze *why* certain models lead to better comprehension outcomes by inspecting each question's:

* **Utility**: Contribution to a simulated learner’s performance
* **Saliency**: Relevance and centrality to the passage
* **Expected Information Gain (EIG)**: Reduction in uncertainty after reading the answer

---

### ⚙️ Options

| Argument             | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `--qa_file`          | Path to the generated QA pairs file (from `run_inference.py`)          |
| `--model_name`       | Model used for utility simulation and scoring (default: `gpt-4o-mini`) |
| `--include_saliency` | If set, computes saliency scores via LLM                               |
| `--include_eig`      | If set, computes Expected Information Gain (EIG) via LLM logprobs      |
| `--output_dir`       | Directory to save the evaluation results (default: `q_metrics/`)       |

---

### 📥 Input Format

Each input file should be a `.jsonl` file with entries like:

```json
{
  "subject": "chemistry",
  "chapter": "m50984",
  "section": 3,
  "question": "Why is the atomic radius smaller across a period?",
  "answer": "Because the increased nuclear charge pulls electrons closer to the nucleus."
}
```

These are typically generated from:

```text
output/{model_name}/{subject}_{mode}_qa_pairs.jsonl
```

---

### 📤 Output

Evaluated metrics are saved in a single consolidated file:

```text
q_metrics/{subject}_{mode}_question_metrics.jsonl
```

Each entry includes:

```json
{
  "subject": "chemistry",
  "chapter": "m50984",
  "section": 3,
  "question": "Why is the atomic radius smaller across a period?",
  "answer": "Because the increased nuclear charge pulls electrons closer to the nucleus.",
  "utility": 0.41,
  "saliency": 4,
  "eig": 0.72
}
```

If `--include_saliency` or `--include_eig` is not used, the corresponding fields will be omitted.

---

### ✅ Example Usage

Evaluate utility only:

```bash
python run_eval.py \
  --qa_file output/gpt-4o-mini/chemistry_fewshot_qa_pairs.jsonl
```

Evaluate utility + saliency:

```bash
python run_eval.py \
  --qa_file output/gpt-4o-mini/chemistry_fewshot_qa_pairs.jsonl \
  --include_saliency
```

Evaluate utility + saliency + EIG:

```bash
python run_eval.py \
  --qa_file output/gpt-4o-mini/chemistry_fewshot_qa_pairs.jsonl \
  --include_saliency \
  --include_eig
```
