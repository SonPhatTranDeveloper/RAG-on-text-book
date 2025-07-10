from src.rag.base_rag import BaseRAG


class MultipleChoiceWrapper:
    def __init__(self, rag: BaseRAG):
        self.rag = rag

    def get_answer(self, question: str) -> str:
        """
        Wrap question with multiple choice format.
        """
        question = f"""
            Response with only a single letter (A, B, C, D) as the answer.
            Question: {question}
            Answer:
        """
        return self.rag.get_answer(question)
