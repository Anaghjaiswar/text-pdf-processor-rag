# RAG Document Q&A System

A stateless, lightweight, state-of-the-art Document Q&A system powered by **Llama 3 (via Groq)** and local **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**. It features a fully self-contained, glassmorphic frontend with real-time text streaming.

---

## 🏗️ System Architecture

The project splits functionality into two isolated pipelines: **Ingestion** and **Retrieval / Inference**.

```mermaid
graph LR
    %% Ingestion Flow Customization
    subgraph Ingestion_Pipeline ["📥 Ingestion Pipeline"]
        A(Upload PDF) --> B[Docling Converter]
        B -->|Markdown text| C[Markdown Header Splitter]
        C -->|Header boundaries| D[Safety Recursive Splitter]
        D -->|Semantic Chunks| E[Shared Embeddings Model]
        E -->|all-MiniLM-L6-v2| F[(FAISS Vector Store)]
    end

    %% Query Flow Customization
    subgraph Query_Pipeline ["⚡ Retrieval & Query Pipeline"]
        G(User Query) --> H[Shared Embeddings Model]
        H -->|Query Vector| I[(FAISS Vector Store)]
        I -->|Similarity Search| J[Retrieve Top-K Chunks]
        J -->|Context String| K[Prompt Builder]
        L(Chat History) --> K
        K -->|Context + History| M[Llama 3.3 via Groq]
        M -->|Streaming Tokens| N(Glassmorphic UI)
    end

    %% Premium Glassmorphism Styling Configurations
    style Ingestion_Pipeline fill:#0f172a,stroke:#4f46e5,stroke-width:2px,color:#ffffff
    style Query_Pipeline fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#ffffff
    
    style A fill:#1e1b4b,stroke:#6366f1,color:#f3f4f6
    style G fill:#083344,stroke:#22d3ee,color:#f3f4f6
    style N fill:#064e3b,stroke:#10b981,color:#f3f4f6
    
    style F fill:#022c22,stroke:#10b981,stroke-width:2px,color:#ffffff
    style I fill:#022c22,stroke:#10b981,stroke-width:2px,color:#ffffff
    
    classDef default fill:#1e293b,stroke:#334155,font-weight:500,color:#cbd5e1;
    class B,C,D,E,H,J,K,M,L default;
```

---

## 📦 Folder Structure

The repository structure shows both the root files and the core project package contents:

```text
gen-ai/                         # Root repository folder
├── rag-document-q&a-project/     # Core RAG project directory
│   ├── __init__.py               # Package initializer exposing core engines
│   ├── doc_processing_engine.py   # Document ingestion, splitting, and vectorizing
│   ├── llm_engine.py             # Context retrieval and LLM reasoning chain
│   ├── rag_pipeline.py           # Orchestrator coordinating ingestion and QA
│   ├── main.py                   # FastAPI server exposing the endpoints
│   ├── index.html                # Custom CSS glassmorphic streaming UI
│   └── README.md                 # Project guide and documentation
├── requirements.txt            # Shared Python dependencies list
├── .gitignore                  # Git ignore rules
├── cnn -deeplearning-part-2.pdf    # Test PDF file
```

### Module Descriptions:

*   **`doc_processing_engine.py`**:
    Parses PDF files into structured layouts (headers, tables, lists) using the `Docling` library. It splits the document structurally by Markdown headers to keep contextual boundaries, runs a secondary recursive splitting for safety limits, and generates local embeddings to save in a `FAISS` database.
*   **`llm_engine.py`**:
    Handles retrieving the relevant text chunks, setting up the RAG system prompt with memory, and building the LangChain Expression Language (LCEL) chain. It contains a modular history trimmer that slices the chat history to the last 2 turns (4 messages) to avoid token bloat.
*   **`rag_pipeline.py`**:
    Acts as the orchestrator class. It coordinates the ingestion of documents and queries, forwarding them to their respective sub-engines. It also manages stateful chat history in the pipeline instance.
*   **`main.py`**:
    Exposes the stateless HTTP endpoints. It utilizes the **Singleton Pattern** to load the HuggingFace embedding model weights globally once on startup, preventing the 1-3 seconds reloading latency on queries.
*   **`index.html`**:
    A zero-framework web client that secures the user's Groq API key in local storage, handles file selection and uploading, and consumes streaming responses chunk-by-chunk using raw JS streams.

---

## 🔄 Interaction Sequence Flowchart

```mermaid
sequenceDiagram
    autonumber
    actor User as User (Browser)
    participant API as FastAPI Server (main.py)
    participant Pipeline as RagPipeline
    participant DB as FAISS Database
    participant Groq as Groq Cloud API

    %% Ingestion
    Note over User, API: 1. Ingestion Phase
    User->>API: POST /ingest (multipart/form PDF)
    Note over API: Parse PDF, Chunk & Vectorize<br/>(Runs on shared embeddings)
    API->>DB: Save Index to disk (/faiss_db)
    API-->>User: JSON Response (Success)

    %% Querying
    Note over User, API: 2. Query Phase
    User->>API: POST /query (Question + X-Groq-API-Key Header)
    API->>Pipeline: Instantiate pipeline (with shared embeddings)
    Pipeline->>DB: Similarity search for Question
    DB-->>Pipeline: Return Top-K Text Chunks
    Pipeline->>API: Return Stream Generator
    API-->>User: HTTP StreamingResponse (Chunk-by-chunk)
    Groq-->>API: Stream tokens (LLM Generation)
    API-->>User: Stream Text Output to Chat Bubble
```

---

## ⚡ Key Optimizations

1.  **Shared Embeddings Singleton:**
    Loading a local PyTorch model (`all-MiniLM-L6-v2`) on every request causes massive CPU/RAM bottlenecks. `main.py` loads the model once globally and passes the reference (`shared_embeddings`) to the engines. Queries are resolved in under 100ms.
2.  **Stateless API Design:**
    The API does not store API keys or persistent user histories. The API keys are provided by the client, and chat history is sent from the frontend/caller or discarded, allowing the server to remain lightweight and scale horizontally.
3.  **Active Memory Trimmer:**
    LangChain LCEL uses a custom `trim_history` filter method. It slices message collections dynamically to keep only the last 2 turns (4 messages), ensuring the LLM is context-grounded without running out of token limits.

---

## 🚀 Getting Started

Follow these steps to clone the repository, set up your environment, install dependencies, and boot the server.

### 1. Clone the Repository & Navigate

Clone the repository and enter the root directory:

```bash
git clone https://github.com/Anaghjaiswar/text-pdf-processor-rag.git
cd gen-ai
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment to manage dependencies locally:

*   **Windows (PowerShell):**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
*   **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

### 3. Install Dependencies

Install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

*(Note: We use the lightweight `sentence-transformers` library to run embedding calculations on CPU, making it fully zero-setup without requiring Ollama running locally).*

### 4. Run the Project

1.  **Start the FastAPI backend server:**
    Navigate into the project directory and run `main.py`:
    ```bash
    cd rag-document-q&a-project
    python main.py
    ```
    The server will boot and listen on `http://localhost:8000`.

2.  **Open the Frontend Client:**
    Double-click the `index.html` file inside the `rag-document-q&a-project/` folder to open the chat interface directly in your browser.

3.  **Discuss Your Document:**
    *   Enter your Groq API Key (`gsk_...`) in the sidebar.
    *   Select/drag-and-drop a PDF file and click **Ingest Document**.
    *   Start asking questions about the document in the chat interface!
