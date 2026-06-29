import os
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from doc_processing_engine import DocProcessingEngine
from rag_pipeline import RagPipeline
from langchain_huggingface import HuggingFaceEmbeddings


app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the embeddings model globally on startup (Singleton pattern)
shared_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_processor = DocProcessingEngine(embeddings=shared_embeddings)


@app.get("/status")
async def get_status():
    """Checks if the FAISS vector database is already built and contains index files."""
    faiss_index_path = os.path.join("./faiss_db", "index.faiss")
    db_exists = os.path.exists(faiss_index_path)
    return {
        "db_exists": db_exists,
        "message": "Database is ready" if db_exists else "No database found"
    }


class QueryRequest(BaseModel):
    question: str
    stream: bool = True


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Uploads and ingests a PDF document to build/update the vector database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("./temp_uploads", exist_ok=True)
    temp_file_path = os.path.join("./temp_uploads", file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_processor.process_pdf(temp_file_path)
        return {"status": "success", "message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/query")
async def query_rag(
    request: QueryRequest,
    x_groq_api_key: str = Header(..., description="Your Groq API Key"),
):
    """Queries the RAG pipeline using the provided Groq API key."""
    if not x_groq_api_key.strip():
        raise HTTPException(status_code=400, detail="X-Groq-API-Key header cannot be empty.")

    try:
        # Reuse the globally loaded embeddings object
        pipeline = RagPipeline(groq_api_key=x_groq_api_key, embeddings=shared_embeddings)

        if request.stream:
            return StreamingResponse(
                pipeline.query_stream(request.question, chat_history=[]),
                media_type="text/plain",
            )
        else:
            answer = pipeline.query(request.question, chat_history=[])
            return {"answer": answer}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
