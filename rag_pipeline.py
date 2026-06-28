import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings


from doc_processing_engine import DocProcessingEngine
from llm_engine import LLMEngine


class RagPipeline:
    """A pipeline that orchestrates Document Processing and LLM QA reasoning.

    This class coordinates the document ingestion phase (converting PDFs, chunking,
    and generating/storing FAISS vector embeddings) and the execution phase
    (retrieving context and generating answers).
    """

    def __init__(
        self,
        groq_api_key: str,
        faiss_storage_path: str = "./faiss_db",
        model_name: str = "llama-3.3-70b-versatile",
        model_provider: str = "groq",
        temperature: float = 0.3,
        prompt: Optional[ChatPromptTemplate] = None,
        max_conversations_to_keep: int = 2,
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ) -> None:
        """Initializes the pipeline, creating instances of both engines."""
        self.faiss_storage_path = faiss_storage_path
        self.doc_processor = DocProcessingEngine(
            faiss_storage_path=faiss_storage_path,
            embeddings=embeddings,
        )
        self.llm_engine = LLMEngine(
            groq_api_key=groq_api_key,
            faiss_storage_path=faiss_storage_path,
            model_name=model_name,
            model_provider=model_provider,
            temperature=temperature,
            prompt=prompt,
            max_conversations_to_keep=max_conversations_to_keep,
            embeddings=embeddings,
        )
        self.chat_history: List[BaseMessage] = []

    def ingest_document(self, pdf_path: str, force_rebuild: bool = False) -> None:
        """Processes a PDF document and updates/builds the FAISS vector database."""
        db_exists = os.path.exists(os.path.join(self.faiss_storage_path, "index.faiss"))
        if db_exists and not force_rebuild:
            return

        self.doc_processor.process_pdf(pdf_path)

    def query(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
        k: int = 4,
    ) -> str:
        """Queries the RAG pipeline, manages chat history state, and generates an answer."""
        history = self.chat_history if chat_history is None else chat_history
        
        answer = self.llm_engine.generate_answer(question, chat_history=history, k=k)
        
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        
        return answer

    def query_stream(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
        k: int = 4,
    ):
        """Queries the RAG pipeline, yields answer chunks, and appends the completed turn to the history."""
        history = self.chat_history if chat_history is None else chat_history
        
        full_answer = []
        for chunk in self.llm_engine.stream_answer(question, chat_history=history, k=k):
            full_answer.append(chunk)
            yield chunk
            
        complete_response = "".join(full_answer)
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=complete_response))

    def clear_history(self) -> None:
        """Clears the internal conversation history state."""
        self.chat_history.clear()
