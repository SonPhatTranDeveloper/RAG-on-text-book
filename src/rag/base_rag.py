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
from llama_index.core.query_engine import SubQuestionQueryEngine, RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)


class BaseRAG(ABC):
    def __init__(self):
        """
        Initialize the RAG
        """
        self.llm = self.get_llm()
        self.query_engine = self.get_query_engine()

    @abstractmethod
    def get_llm(self) -> any:
        """
        Get the LLM model to use for the RAG.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_query_engine(self) -> any:
        """
        Get the query engine to use for the RAG.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_answer(self, question: str) -> str:
        """
        Get the answer for the question.
        """
        raise NotImplementedError("Subclasses must implement this method")


class LocalRAG(BaseRAG):
    def __init__(
        self, storage_dir: str = "src/.index_storage", data_dir: str = "../data"
    ):
        super().__init__()
        self.storage_dir = storage_dir
        self.data_dir = data_dir

    def get_llm(self) -> any:
        """
        Use default OpenAI model.
        """
        return OpenAI(model="gpt-4o-mini")

    def get_query_engine(self) -> any:
        """
        Load local index from storage and return query engine.
        """
        # Check if index already exists
        if os.path.exists(self.storage_dir):
            logger.info("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new index...")
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=self.storage_dir)

        return index.as_query_engine()


class LocalBaselineRAG(LocalRAG):
    def get_answer(self, question: str) -> str:
        return self.query_engine.query(question).response


class LocalSubQuestionRAG(LocalRAG):
    def get_query_engine(self) -> any:
        """
        Get the SubQuestionQueryEngine to use for the RAG.
        """
        # Create query engine tools
        query_engine_tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name="document_search",
                description="Công cụ này để tìm kiếm thông tin trong sách giáo khoa cấp trung học phổ thông.",
            )
        ]
        self.query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools, llm=self.llm
        )

    def get_answer(self, question: str) -> str:
        return self.query_engine.query(question).response


class LocalReRankRAG(LocalRAG):
    def __init__(
        self,
        storage_dir: str = "src/.index_storage",
        data_dir: str = "src/data",
        bm25_dir: str = "src/.bm25_index_storage",
        embed_similarity_top_k: int = 10,
        bm25_similarity_top_k: int = 5,
        rerank_similarity_top_k: int = 5,
        chunk_size: int = 256,
        chunk_overlap: int = 20,
        num_queries: int = 4,
    ):
        self.embed_dir = storage_dir
        self.data_dir = data_dir
        self.bm25_dir = bm25_dir
        self.embed_similarity_top_k = embed_similarity_top_k
        self.bm25_similarity_top_k = bm25_similarity_top_k
        self.rerank_similarity_top_k = rerank_similarity_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_queries = num_queries
        super().__init__()

    def _get_vector_retriever(self) -> any:
        """
        Get the vector-based retriever to use for the RAG.
        """
        # Get vector-based retriever
        # Check if index already exists
        if os.path.exists(self.embed_dir):
            logger.info("Loading existing vector index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.embed_dir)
            index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new vector index...")
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=self.embed_dir)

        return index.as_retriever(similarity_top_k=self.embed_similarity_top_k)

    def _get_bm25_retriever(self) -> any:
        """
        Get the BM25 retriever to use for the RAG.
        """
        if os.path.exists(self.bm25_dir):
            logger.info("Loading existing BM25 index...")
            bm25_retriever = BM25Retriever.from_persist_dir(path=self.bm25_dir)
        else:
            logger.info("Creating new BM25 index...")
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            node_parser = SentenceSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes, similarity_top_k=self.embed_similarity_top_k
            )
            bm25_retriever.persist(path=self.bm25_dir)

        return bm25_retriever

    def get_query_engine(self) -> any:
        """
        Get the query engine to use for the RAG.
        """
        # Get the vector and BM25 retriever
        vector_retriever = self._get_vector_retriever()
        bm25_retriever = self._get_bm25_retriever()

        # Create query
        query_gen_prompt = (
            "Bạn là một trợ lý hữu ích tạo ra nhiều truy vấn tìm kiếm dựa trên một "
            "truy vấn đầu vào duy nhất bằng tiếng Việt. Tạo {num_queries} truy vấn tìm kiếm bằng tiếng Việt, "
            "mỗi truy vấn trên một dòng, "
            "liên quan đến truy vấn đầu vào sau:\n"
            "Truy vấn: {query}\n"
            "Các truy vấn:\n"
        )

        # Get hybrid retriever
        hybrid_retriever = QueryFusionRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            query_gen_prompt=query_gen_prompt,
            num_queries=self.num_queries,
            llm=self.llm,
            mode="reciprocal_rerank",
            similarity_top_k=self.rerank_similarity_top_k,
        )

        return RetrieverQueryEngine.from_args(hybrid_retriever)

    def get_answer(self, question: str) -> str:
        return self.query_engine.query(question).response
