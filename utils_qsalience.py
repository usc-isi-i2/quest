# referenced from https://github.com/ritikamangla/QSalience/blob/main/code/predict_salience.py

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
import torch
import re
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_qlora_model(model_name, qlora_model):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(qlora_model)
    model = PeftModel.from_pretrained(base_model, qlora_model)
    return model, tokenizer

def evaluate_model(dataset, model, tokenizer, max_length=4096, model_name=""):
    numeric_preds = []

    for item in dataset:
        system_prompt = "<s>### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        input_text = """### Instruction: {instruction}\n\n### Input: {input}\n\n### Response: """.format(instruction=item['instruction'], input=item['input'])
        inputs = tokenizer.encode(system_prompt + input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model.generate(inputs, max_length=max_length, do_sample=False)

        for output in outputs:
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            match = re.search(r'Response:[\s\S]*?(\d+)', output_text)
            if match:
                prediction = int(match.group(1))
            else:
                prediction = -1  # Default value for cases where the pattern is not found

            numeric_preds.append(prediction)


    df = pd.DataFrame({"prediction": numeric_preds})
    save_name = model_name.split("/")[-1]
    df.to_csv(f"{save_name}_predictions.csv", index=False)

    
class QSalience:
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", qlora_model="lingchensanwen/mistral-ins-generation-best-balanced"):
        self.model, self.tokenizer = merge_qlora_model(model_name, qlora_model)
        self.model.to(device)
        
    def predict_salience(self, article, question):
        system_prompt = "<s>### System:\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        
        input_data = {
            "article": article,
            "question": question
        }
        
        instruction = """
Give a score from 1 to 5 for how important it is for the question to be answered later in the article.
Score = 1 means the question is completely unrelated to the topic of the article.
Score = 2 means the question is related to the article but answering it is not useful in making the article feel complete.
Score = 3 means the question is related to the article but answering it might not enhance the understanding of the article.
Score = 4 means the question is related to the article and answering it is somewhat useful in enhancing the understanding of the article.
Score = 5 means the question is related to the article and should definitely be answered because it might provide explanation for some new concepts.
"""
        input_text = """### Instruction: {instruction}
        
### Input: {input_data}

### Response: """.format(instruction=instruction, input_data=input_data)
        
        inputs = self.tokenizer.encode(system_prompt + input_text, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.model.generate(inputs, max_length=4096, do_sample=False)
        
        for output in outputs:
            output_text = self.tokenizer.decode(output, skip_special_tokens=True)
            match = re.search(r'Response:[\s\S]*?(\d+)', output_text)
            if match:
                prediction = int(match.group(1))
            else:
                prediction = -1  # Default value for cases where the pattern is not found

            return prediction
    
    