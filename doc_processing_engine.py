import os
from typing import List, Tuple, Optional
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


class DocProcessingEngine:
    """Document processing engine for RAG (Retrieval-Augmented Generation) applications.

    This engine orchestrates the document ingestion pipeline. It converts raw PDFs into
    layout-aware Markdown text, splits the content using semantic headers to maintain
    contextual boundaries, performs secondary character-based chunking for safety, and
    saves the generated embeddings into a local FAISS vector store.
    """

    def __init__(
        self,
        faiss_storage_path: str = "./faiss_db",
        embeddings: Optional[HuggingFaceEmbeddings] = None,
    ) -> None:
        """Initializes the DocProcessingEngine with conversion, splitting, and embedding configurations."""
        self.faiss_storage_path: str = faiss_storage_path
        self.converter: DocumentConverter = DocumentConverter()
        
        self.headers_to_split_on: List[Tuple[str, str]] = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3")
        ]
        
        self.markdown_splitter: MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        
        self.safety_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.embeddings: HuggingFaceEmbeddings = embeddings or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def convert_to_markdown(self, pdf_path: str) -> str:
        """Converts a PDF document to a structured Markdown format."""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF file not found: '{pdf_path}'")
        
        try:
            conversion_result = self.converter.convert(pdf_path)
            markdown_text = conversion_result.document.export_to_markdown()
            return markdown_text
        except Exception as e:
            raise RuntimeError(f"Error converting PDF to Markdown for '{pdf_path}': {e}") from e

    def get_structured_docs(self, markdown_text: str) -> List[Document]:
        """Splits Markdown text into semantically cohesive document chunks.

        This method first divides the text based on markdown headers (e.g., # H1, ## H2),
        preserving hierarchical document structure. It then applies a secondary recursive
        character splitter to ensure that each chunk does not exceed token or length safety limits.
        """
        structured_docs = self.markdown_splitter.split_text(markdown_text)
        final_chunks = self.safety_splitter.split_documents(structured_docs)
        return final_chunks

    def create_embeddings_and_save_to_db(self, final_chunks: List[Document]) -> FAISS:
        """Generates vector embeddings for document chunks and persists them to FAISS."""
        db = FAISS.from_documents(final_chunks, self.embeddings)
        db.save_local(self.faiss_storage_path)
        return db

    def process_pdf(self, pdf_path: str) -> FAISS:
        """Processes a PDF file through the entire RAG ingestion pipeline.
        This orchestrator converts the PDF to Markdown, chunks it structure-aware,
        generates text embeddings, and saves the resulting FAISS index locally.
        """
        markdown_text = self.convert_to_markdown(pdf_path)
        final_chunks = self.get_structured_docs(markdown_text)
        db = self.create_embeddings_and_save_to_db(final_chunks)
        return db
