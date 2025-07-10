from src.rag.base_rag import BaseRAG


class MultipleChoiceWrapper:
    def __init__(self, rag: BaseRAG):
        self.rag = rag

    def get_answer(self, question: str) -> str:
        """
        Wrap question with multiple choice format.
        """
        question = f"""
            Hãy chỉ trả lời câu hỏi Multiple choice sau bằng một trong các đáp án A, B, C, D.   
            Câu hỏi: {question}
            Đáp án:
        """
        return self.rag.get_answer(question)
