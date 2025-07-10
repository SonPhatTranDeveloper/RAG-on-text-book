from typing import Optional, Any, List
from llama_index.core.evaluation import EvaluationResult


class MCQEvaluator:
    """
    Custom evaluator for multiple-choice questions that strictly checks if
    the model's response is an exact, case-sensitive match to one of the
    provided reference letters (e.g., "A", "B", "C", "D").
    No processing (stripping, lowercasing, or extracting) is performed on the response.
    """

    def evaluate(
        self,
        query: str,
        response: Optional[str] = None,
        reference_answer: Optional[
            str
        ] = None,  # The single correct letter (e.g., "A", "B")
        reference_answers: Optional[
            List[str]
        ] = None,  # List of correct letters if multiple are valid
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluates the response for a strict exact match against reference answer letters.

        Args:
            query (str): The original query.
            response (Optional[str]): The generated response from the LLM.
            reference_answer (Optional[str]): A single correct letter string.
            reference_answers (Optional[List[str]]): A list of possible correct letter strings.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            EvaluationResult: The result of the strict exact match evaluation.
        """
        if response is None:
            return EvaluationResult(
                query=query,
                response="No response provided.",
                score=0,
                feedback="No response to evaluate.",
                passing=False,
            )

        if reference_answer is None and reference_answers is None:
            return EvaluationResult(
                query=query,
                response=response,
                score=0,
                feedback="No reference answer(s) provided for evaluation.",
                passing=False,
            )

        # The model's response is used as-is, without any processing
        model_response_as_is = response[0].upper()

        # Prepare the reference answer(s) (also used as-is)
        possible_answers: List[str] = []
        if reference_answer:
            possible_answers.append(reference_answer)
        if reference_answers:
            possible_answers.extend(reference_answers)

        # Perform the strict exact match check
        is_exact_match = model_response_as_is in possible_answers
        score = 1.0 if is_exact_match else 0.0
        feedback = (
            f"Response '{model_response_as_is}' "
            f"{'strictly matched' if is_exact_match else 'did not strictly match'} "
            f"one of the expected options: {list(set(possible_answers))}"
        )

        return EvaluationResult(
            query=query,
            response=response,
            score=score,
            feedback=feedback,
            passing=is_exact_match,
        )

    async def aevaluate(self, *args, **kwargs) -> EvaluationResult:
        """Asynchronous evaluation method, which calls the synchronous one."""
        return self.evaluate(*args, **kwargs)
