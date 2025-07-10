from abc import ABC, abstractmethod
from llama_index.llms.openai import OpenAI
import os
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    SimpleDirectoryReader,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine

logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    def __init__(self):
        """
        Initialize the RAG
        """
        self.llm = self.get_llm()
        self.index = self.get_index()

    @abstractmethod
    def get_llm(self) -> any:
        """
        Get the LLM model to use for the RAG.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_index(self) -> any:
        """
        Get the index to use for the RAG.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_answer(self, question: str) -> str:
        """
        Get the answer for the question.
        """
        raise NotImplementedError("Subclasses must implement this method")


class LocalRAG(BaseRAG):
    def get_llm(self) -> any:
        """
        Use default OpenAI model.
        """
        return OpenAI(model="gpt-4o-mini")

    def get_index(self) -> any:
        """
        Load local index from storage.
        """
        storage_dir = "src/.index_storage"

        # Check if index already exists
        if os.path.exists(storage_dir):
            logger.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new index...")
            documents = SimpleDirectoryReader("../data").load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=storage_dir)

        return index


class LocalBaselineRAG(LocalRAG):
    def __init__(self):
        super().__init__()
        self.query_engine = self.index.as_query_engine()

    def get_answer(self, question: str) -> str:
        return self.query_engine.query(question).response


class LocalSubQuestionRAG(LocalRAG):
    def __init__(self):
        super().__init__()

        # Create query engine tools
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.index.as_query_engine(),
                name="document_search",
                description="Công cụ này để tìm kiếm thông tin trong sách giáo khoa cấp trung học phổ thông.",
            )
        ]
        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools, llm=self.llm
        )

    def get_answer(self, question: str) -> str:
        return self.query_engine.query(question).response
