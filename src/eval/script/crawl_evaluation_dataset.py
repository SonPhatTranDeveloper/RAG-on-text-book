import os
import json
import logging
import argparse
from src.eval.pipeline import create_standard_pipeline
from dotenv import load_dotenv
from src.eval.pipeline.base import BasePipeline

# Set up logging to both file and console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Capture all levels

# File handler
file_handler = logging.FileHandler("pipeline-run.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if script is rerun
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def process_single_subject(
    pipeline: BasePipeline, subject_path: str, subject_name: str
) -> None:
    """
    Processes a single subject folder: reads URLs, processes them with a pipeline,
    and saves all results from all URLs into a single JSON file named 'dataset.json'
    within the subject folder.

    Args:
        pipeline (BasePipeline): The pipeline instance to process URLs.
        subject_path (str): The full path to the subject directory.
        subject_name (str): The name of the subject (e.g., 'subject_1').

    Returns:
        None
    """
    logger.info(f"\n--- Processing subject: {subject_name} ---")
    urls_file_path = os.path.join(subject_path, "urls.txt")

    # List to accumulate all results for the current subject
    all_subject_results = []
    processed_urls_count = 0

    # 1. Read URLs from urls.txt
    if os.path.exists(urls_file_path):
        logger.info(f"  Reading URLs from: {urls_file_path}")
        with open(urls_file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]  # Read non-empty lines
        logger.info(f"  Found {len(urls)} URLs.")

        # 2. Process each URL using the pipeline and accumulate results
        for i, url in enumerate(urls):
            try:
                results = pipeline.run(url)
                if (
                    isinstance(results, list) and results
                ):  # Ensure it's a non-empty list
                    all_subject_results.extend(results)  # Add results to the main list
                    logger.info(
                        f"    Added {len(results)} items from URL '{url}' to subject's dataset."
                    )
                    processed_urls_count += 1
                elif isinstance(results, list) and not results:
                    logger.warning(
                        f"    Pipeline for '{url}' returned an empty list. No data to add."
                    )
                else:
                    logger.warning(
                        f"    Pipeline for '{url}' did not return a list. Skipping adding its data."
                    )
            except Exception as e:
                logger.error(f"    Error processing URL '{url}': {e}")
    else:
        logger.info(f"  No 'urls.txt' found in {subject_path}. Skipping subject.")
        return  # Exit the function for this subject

    # 3. Save all accumulated results into a single JSON file for the subject
    if all_subject_results:
        # Define a single output filename as 'dataset.json'
        output_filename = "dataset.json"
        dataset_file_path = os.path.join(subject_path, output_filename)

        logger.info(
            f"  Saving consolidated dataset ({len(all_subject_results)} items) for subject '{subject_name}' to: {dataset_file_path}"
        )
        with open(dataset_file_path, "w", encoding="utf-8") as f:
            json.dump(all_subject_results, f, indent=4)
        logger.info(
            f"  Successfully processed and saved consolidated data for {processed_urls_count} URLs in {subject_name}."
        )
    else:
        logger.info(
            f"  No data generated for any URL in {subject_name}. No consolidated file saved."
        )


def process_subjects_and_generate_datasets(
    pipeline: BasePipeline, grade_directory: str, subject_name: str = None
) -> None:
    """
    Loops through each subject folder, and calls process_single_subject for each.
    If subject_name is provided, only processes that specific subject.

    Args:
        pipeline (BasePipeline): The pipeline instance to process URLs.
        grade_directory (str): The path to the main 'grade' directory.
        subject_name (str, optional): Specific subject to process. If None, processes all subjects.

    Returns:
        None
    """
    if not os.path.exists(grade_directory):
        logger.error(f"Error: Grade directory '{grade_directory}' not found.")
        return

    if subject_name:
        # Process only the specified subject
        subject_path = os.path.join(grade_directory, subject_name)
        if os.path.isdir(subject_path):
            logger.info(f"Processing specific subject: {subject_name}")
            process_single_subject(pipeline, subject_path, subject_name)
        else:
            logger.error(
                f"Error: Subject directory '{subject_name}' not found in '{grade_directory}'."
            )
            return
    else:
        # Process all subjects
        logger.info(f"Starting processing in '{grade_directory}'...")

        # Iterate through all items in the grade directory
        for subject_name in os.listdir(grade_directory):
            subject_path = os.path.join(grade_directory, subject_name)

            # Check if it's a directory (i.e., a subject folder)
            if os.path.isdir(subject_path):
                process_single_subject(pipeline, subject_path, subject_name)
            else:
                logger.info(
                    f"Skipping non-directory item: {subject_name} in {grade_directory}"
                )

    logger.info("\nProcessing complete!")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Process URLs from subject folders and generate datasets."
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default="vietjack",
        help="The pipeline to use. Currently supported: 'vietjack'.",
    )
    parser.add_argument(
        "--grade_dir",
        "-g",
        type=str,
        default="grade",  # Default value if not provided
        help="The path to the main 'grade' directory containing subject folders.",
    )
    parser.add_argument(
        "--subject",
        "-s",
        type=str,
        default=None,
        help="Specific subject to process. If not provided, processes all subjects.",
    )
    args = parser.parse_args()
    grade_directory = args.grade_dir
    subject_name = args.subject
    pipeline = create_standard_pipeline(args.pipeline)
    process_subjects_and_generate_datasets(pipeline, grade_directory, subject_name)
