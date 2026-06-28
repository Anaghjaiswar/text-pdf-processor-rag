import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chat_models import init_chat_model

load_dotenv()


if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "RAG-MVP-Project")


class LLMEngine:
    """Core LLM reasoning engine for RAG (Retrieval-Augmented Generation) applications.

    This engine handles retrieval of relevant context from a local FAISS vector store,
    constructs formatted prompts containing the retrieved context, and invokes a chat
    model to generate precise, grounded answers.
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
        """Initializes the LLMEngine with FAISS storage, Chat model, and prompt configurations."""
        self.faiss_storage_path: str = faiss_storage_path
        self.max_conversations_to_keep: int = max_conversations_to_keep
        self.embeddings: HuggingFaceEmbeddings = embeddings or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if not groq_api_key:
            raise ValueError("groq_api_key must be provided and cannot be empty.")

        os.environ["GROQ_API_KEY"] = groq_api_key

        self.llm = init_chat_model(
            model_name,
            model_provider=model_provider,
            temperature=temperature,
            api_key=groq_api_key,
        )

        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant for question-answering tasks.\n"
                        "Answer the question using ONLY the retrieved context below. "
                        "If you don't know the answer or if the context does not contain "
                        "the answer, state clearly that the answer is not available "
                        "in the provided document. Do not make up facts.\n\n"
                        "Context:\n{context}",
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.trim_history(x["chat_history"])
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def trim_history(self, chat_history: List[BaseMessage]) -> List[BaseMessage]:
        """Trims chat history to retain at most the last N conversations (turns)."""
        limit = self.max_conversations_to_keep * 2
        if len(chat_history) <= limit:
            return chat_history
        return chat_history[-limit:]

    def get_relevant_chunks(self, question: str, k: int = 4) -> List[Document]:
        """Retrieves the most semantically relevant document chunks from the FAISS database."""
        faiss_index_path = os.path.join(self.faiss_storage_path, "index.faiss")
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(
                f"Couldn't find vector database index at '{self.faiss_storage_path}'. "
                "Please ingest documents first using the document processing engine."
            )

        db = FAISS.load_local(
            self.faiss_storage_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        relevant_chunks = db.similarity_search(question, k=k)
        return relevant_chunks

    def generate_answer(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
        k: int = 4,
    ) -> str:
        """Retrieves context and generates a grounded answer for the given question."""
        try:
            context_docs = self.get_relevant_chunks(question, k=k)
            context_str = "\n\n".join([doc.page_content for doc in context_docs])

            return self.chain.invoke(
                {
                    "context": context_str,
                    "question": question,
                    "chat_history": chat_history or [],
                }
            )
        except Exception as e:
            raise RuntimeError(f"Error generating answer for question '{question}': {e}") from e

    def stream_answer(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
        k: int = 4,
    ):
        """Retrieves context and streams the generated answer for the given question."""
        try:
            context_docs = self.get_relevant_chunks(question, k=k)
            context_str = "\n\n".join([doc.page_content for doc in context_docs])

            yield from self.chain.stream(
                {
                    "context": context_str,
                    "question": question,
                    "chat_history": chat_history or [],
                }
            )
        except Exception as e:
            raise RuntimeError(f"Error streaming answer for question '{question}': {e}") from e
