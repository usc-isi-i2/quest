#!/bin/bash

input_dir="output/gpt-4o-mini/"
output_dir="q_metrics/"
subjects=(sociology chemistry us-history microbiology economics)
for subject in ${subjects[@]}; do

    fn=${input_dir}${subject}_default_qa_pairs.jsonl
    outputname=${output_dir}${subject}_default_question_metrics.jsonl
    # if outputname does not exist, run eval
    if [ ! -f q_metrics/${outputname} ]; then
        python run_eval.py --qa_file $fn --include_saliency_zeroshot --include_eig
    fi

    fn=${input_dir}${subject}_fewshot_qa_pairs.jsonl
    outputname=${output_dir}${subject}_fewshot_question_metrics.jsonl
    # if outputname does not exist, run eval
    if [ ! -f q_metrics/${outputname} ]; then
        python run_eval.py --qa_file $fn  --include_saliency_zeroshot --include_eig
    fi
done

