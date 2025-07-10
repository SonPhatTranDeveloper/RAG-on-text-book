#!/bin/bash

# This script automates the execution of the MCQ evaluation script for various subjects.
# It uses a loop to iterate through a predefined list of subjects, making the script
# more concise, readable, and easier to maintain.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration Variables ---
# Base command for the evaluation script
BASE_CMD="uv run python src/eval/script/run_mcq_eval.py"

# Common arguments for the evaluation script
RAG_TYPE="subquestion"
GRADE="grade_10"
MAX_QUESTIONS=100
OUTPUT_DIR="src/eval_result/baseline"

# List of subjects to evaluate
SUBJECTS=(
    "dia_ly"
    "hoa_hoc"
    "lich_su"
    "toan"
    "vat_ly"
    "ngu_van"
    "sinh_hoc"
)

echo "Starting MCQ evaluation for grade ${GRADE}..."
echo "Output files will be saved in: ${OUTPUT_DIR}"
echo "--------------------------------------------------"

# Loop through each subject and run the evaluation command
for SUBJECT in "${SUBJECTS[@]}"; do
    OUTPUT_FILE="${OUTPUT_DIR}/${SUBJECT}.json"
    
    echo "Running evaluation for subject: ${SUBJECT}..."
    
    # Construct and execute the full command
    ${BASE_CMD} \
        --rag-type "${RAG_TYPE}" \
        --grade "${GRADE}" \
        --subject "${SUBJECT}" \
        --max-questions "${MAX_QUESTIONS}" \
        --output-file "${OUTPUT_FILE}"

    echo "Finished evaluation for ${SUBJECT}. Results saved to ${OUTPUT_FILE}"
    echo "--------------------------------------------------"
done

echo "All MCQ evaluations completed successfully!"