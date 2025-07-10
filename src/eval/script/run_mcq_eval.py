#!/usr/bin/env python3
"""
Script to run multiple choice evaluation on RAG systems.

This script demonstrates how to use the MultipleChoiceEvaluationPipeline
to evaluate different RAG systems on multiple choice questions.
"""

import logging
import argparse
import sys
from src.eval.eval_pipeline.multiple_choice_eval import MultipleChoiceEvaluationPipeline
from src.rag.base_rag import LocalBaselineRAG, LocalSubQuestionRAG
from dotenv import load_dotenv


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("mcq_eval")
    logger.setLevel(getattr(logging, level.upper()))

    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)

    return logger


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run multiple choice evaluation on RAG systems"
    )
    parser.add_argument(
        "--rag-type",
        choices=["baseline", "subquestion"],
        default="baseline",
        help="Type of RAG system to evaluate",
    )
    parser.add_argument(
        "--grade",
        default="grade_10",
        help="Grade level to evaluate (default: grade_10)",
    )
    parser.add_argument(
        "--subject",
        choices=[
            "toan",
            "ngu_van",
            "vat_ly",
            "hoa_hoc",
            "sinh_hoc",
            "dia_ly",
            "lich_su",
        ],
        required=True,
        help="Subject to evaluate",
    )
    parser.add_argument(
        "--max-questions", type=int, help="Maximum number of questions to evaluate"
    )
    parser.add_argument(
        "--output-file", help="File to save detailed results (JSON format)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level)

    logger.info("Starting multiple choice evaluation")
    logger.info(f"RAG Type: {args.rag_type}")
    logger.info(f"Grade: {args.grade}")
    logger.info(f"Subject: {args.subject}")
    if args.max_questions:
        logger.info(f"Max Questions: {args.max_questions}")

    try:
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        if args.rag_type == "baseline":
            rag_system = LocalBaselineRAG()
            logger.info("Using LocalBaselineRAG")
        else:
            rag_system = LocalSubQuestionRAG()
            logger.info("Using LocalSubQuestionRAG")

        # Initialize evaluation pipeline
        logger.info("Initializing evaluation pipeline...")
        pipeline = MultipleChoiceEvaluationPipeline(rag_system)

        # Run evaluation
        logger.info(f"Starting evaluation for {args.grade}/{args.subject}")
        results = pipeline.evaluate_dataset(
            grade=args.grade, subject=args.subject, max_questions=args.max_questions
        )

        # Print summary
        pipeline.print_summary(results)

        # Save detailed results if requested
        if args.output_file:
            logger.info(f"Saving detailed results to {args.output_file}")
            import json

            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info("Results saved successfully")

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
