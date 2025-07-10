import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.eval.evaluator import MCQEvaluator
from src.utils.eval import load_eval_dataset, load_eval_dataset_by_grade
from src.rag.base_rag import BaseRAG

logger = logging.getLogger(__name__)


class MultipleChoiceEvaluationPipeline:
    """
    A pipeline for evaluating RAG systems on multiple choice questions.

    This pipeline:
    1. Loads multiple choice question-answer pairs from datasets
    2. Uses a RAG system to generate answers
    3. Evaluates the answers using MCQEvaluator
    4. Calculates accuracy metrics
    """

    def __init__(self, rag: BaseRAG):
        """
        Initialize the evaluation pipeline.

        Args:
            rag: A RAG system that implements the BaseRAG interface
        """
        self.rag = rag
        self.evaluator = MCQEvaluator()
        self.results = []

    def evaluate_dataset(
        self,
        grade: str,
        subject: Optional[str] = None,
        max_questions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG system on a dataset.

        Args:
            grade: Grade level (e.g., "grade_10")
            subject: Subject name (e.g., "toan", "ngu_van"). If None, evaluates all subjects
            max_questions: Maximum number of questions to evaluate. If None, evaluates all

        Returns:
            Dictionary containing evaluation results and metrics
        """
        # Load dataset
        if subject:
            dataset = load_eval_dataset(grade, subject)
        else:
            dataset = load_eval_dataset_by_grade(grade)

        if max_questions:
            dataset = dataset[:max_questions]

        logger.info(
            f"Evaluating {len(dataset)} questions from {grade}"
            + (f"/{subject}" if subject else "")
        )

        # Evaluate each question
        correct_count = 0
        total_count = len(dataset)

        for i, qa_pair in enumerate(tqdm(dataset, desc="Evaluating questions")):
            question = qa_pair["question"]
            correct_answer = qa_pair["answer"]

            try:
                # Get RAG response
                rag_response = self.rag.get_answer(question)

                # Evaluate the response
                eval_result = self.evaluator.evaluate(
                    query=question,
                    response=rag_response,
                    reference_answer=correct_answer,
                )

                # Store result
                result = {
                    "question_id": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "rag_response": rag_response,
                    "is_correct": eval_result.passing,
                    "score": eval_result.score,
                    "feedback": eval_result.feedback,
                }
                self.results.append(result)

                if eval_result.passing:
                    correct_count += 1

            except Exception as e:
                logger.error(f"Error evaluating question {i}: {e}")
                result = {
                    "question_id": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "rag_response": "ERROR",
                    "is_correct": False,
                    "score": 0.0,
                    "feedback": f"Error: {str(e)}",
                }
                self.results.append(result)

        # Calculate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        metrics = {
            "total_questions": total_count,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "grade": grade,
            "subject": subject,
            "rag_system": self.rag.__class__.__name__,
        }

        logger.info(
            f"Evaluation complete. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})"
        )

        return {"metrics": metrics, "results": self.results}

    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """
        Get detailed results from the last evaluation.

        Returns:
            List of detailed result dictionaries
        """
        return self.results

    def log_summary(self, results: Dict[str, Any]):
        """
        Log a summary of evaluation results.

        Args:
            results: Results dictionary from evaluate_dataset
        """
        logger.info("=" * 60)
        logger.info("MULTIPLE CHOICE EVALUATION SUMMARY")
        logger.info("=" * 60)

        metrics = results["metrics"]
        subject_str = f"/{metrics['subject']}" if metrics["subject"] else ""
        logger.info(f"Dataset: {metrics['grade']}{subject_str}")
        logger.info(f"RAG System: {metrics['rag_system']}")
        logger.info(
            f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_answers']}/{metrics['total_questions']})"
        )

        logger.info("=" * 60)

