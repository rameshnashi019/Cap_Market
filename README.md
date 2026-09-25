# PostTradeX — AI-Powered Post-Trade Operations

PostTradeX is a Retrieval-Augmented Generation (RAG) project designed for post-trade operations and settlement workflows. It helps teams retrieve accurate answers from internal operational documents, policies, exception notes, and operational procedures without manually searching across fragmented sources.

The project uses a modular architecture to ingest documents, clean and validate them, split them into chunks, embed them into a vector store, and answer natural-language questions grounded in source material.

---

## Business Problem

Post-trade operations teams deal with a large volume of domain knowledge spread across:

- settlement and cut-off rules
- exception handling notes
- EOD and operations procedures
- trade matching requirements
- reconciliation guidance
- operational FAQs and historical notes

This knowledge is often stored in unstructured files and documents, which makes it difficult for analysts and support teams to find the correct answer quickly. In a high-risk environment like post-trade processing, slow or inconsistent retrieval can create delays, operational errors, and compliance issues.

PostTradeX solves this by turning internal documentation into a searchable, AI-powered knowledge assistant.

---

## Project Goal

Build an intelligent post-trade operations assistant that can:

- ingest operational documents and policies
- normalize and validate content quality
- index knowledge in a vector database
- retrieve the most relevant context for a user query
- generate grounded answers using an LLM
- provide a web-based interface for internal users

---

## Core Capabilities

- Document ingestion from local files and operational sources
- Content cleaning and validation before indexing
- Chunking for effective retrieval
- Embedding generation and vector storage with Chroma
- Similarity-based retrieval of relevant evidence
- LLM-based answer generation grounded in source content
- FastAPI-based web app with authentication
- Evaluation-ready pipeline for testing retrieval and answer quality

---

## Architecture Overview

The system follows a standard RAG pipeline:

1. Documents are loaded from the data directory.
2. Content is cleaned and quality-checked.
3. Text is chunked into manageable segments.
4. Chunks are embedded and stored in Chroma.
5. A user question is converted to an embedding and matched against the vector store.
6. The most relevant chunks are retrieved.
7. The LLM generates a grounded answer using the retrieved evidence.

The orchestration is handled by the main pipeline in [src/rag/retrieval/rag_pipeline.py](src/rag/retrieval/rag_pipeline.py).

---

## Repository Structure

```text
Rag/
├── README.md
├── pyproject.toml
├── data/
│   ├── docs/
│   └── vectorstore/
├── logs/
├── src/
│   └── rag/
│       ├── api/
│       │   ├── routes.py
│       │   ├── static/
│       │   └── templates/
│       ├── config/
│       │   └── settings.py
│       ├── data/
│       │   ├── cleaner.py
│       │   ├── chunk_evaluator.py
│       │   ├── chunker.py
│       │   ├── ingestion.py
│       │   ├── loader.py
│       │   ├── metadata.py
│       │   └── validator.py
│       ├── embeddings/
│       │   └── provider.py
│       ├── llm/
│       │   ├── generator.py
│       │   └── prompts.py
│       ├── retrieval/
│       │   ├── rag_pipeline.py
│       │   └── retriever.py
│       ├── vectorstore/
│       │   └── chroma_store.py
│       └── __init__.py
├── tests/
│   ├── test_chroma_store.py
│   ├── test_chunking.py
│   ├── test_document_cleaner.py
│   ├── test_document_ingestion.py
│   ├── test_embeddings.py
│   ├── test_fastapi_app.py
│   ├── test_llm_generation.py
│   ├── test_metadata_enrichment.py
│   ├── test_quality_validator.py
│   ├── test_rag_pipeline.py
│   ├── test_rag_pipeline_indexing.py
│   ├── test_retrieval.py
│   └── ...
└── chroma_db/
```

---

## Technology Stack

- Python
- FastAPI
- ChromaDB
- LangChain
- OpenAI / Hugging Face embeddings
- Sentence Transformers
- Jinja2 templates
- Pytest

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Using pip:

```bash
pip install -e .
```

Or using uv:

```bash
uv sync
```

### 4. Configure environment variables

Create a `.env` file based on the project configuration. The app reads settings such as authentication values and model configuration from environment variables.

Example:

```env
OPENAI_CHAT_MODEL=gpt-4o-mini
APP_USERNAME=admin
APP_PASSWORD=change-me
APP_SECRET_KEY=your-secret-key
```

---

## Running the Application

### Start the API server

```bash
uv run uvicorn rag.api.routes:app --reload
```

Then open:

```text
http://127.0.0.1:8000/
```

The app includes a login screen and a protected chat interface.

---

## Usage

The project exposes a core pipeline object that can be used programmatically:

```python
from rag.retrieval.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.sync_and_index()
answer = pipeline.ask("What are the main stages of EOD processing?", k=4)
print(answer.answer)
```

This workflow:

- syncs source documents into the vector store
- retrieves relevant context
- produces a grounded answer

---

## Example Use Cases

- “What are the mandatory fields for trade matching?”
- “When should a break be closed?”
- “What are the main stages of the post-trade lifecycle?”
- “What happens during end-of-day processing?”

These questions are representative of the operational support the system is designed to answer.

---

## Data and Knowledge Base

The project is built around operational knowledge found in the data folder, including:

- policy and procedural documents
- trade processing notes
- exceptions and historical records
- business rules and operational metadata

These files are indexed and used to power the retrieval layer.

---

## Testing

The project includes a range of tests covering ingestion, cleaning, chunking, embeddings, retrieval, and generation.

Run the full test suite with:

```bash
uv run pytest -q
```

---

## Challenges Addressed

This implementation demonstrates how to solve several real-world engineering problems:

- noisy or inconsistent document quality
- chunking strategies for domain text
- retrieval relevance for operational knowledge
- ensuring answers are grounded in source material
- integrating AI into internal operations workflows
- building a production-like chatbot with authentication

---

## Future Enhancements

Potential next steps include:

- support for additional document sources and connectors
- better metadata filtering for relevant operational domains
- evaluation metrics for answer quality and retrieval precision
- user role-based access control
- audit logs for traceability
- deployment packaging for cloud environments

---

## Conclusion

PostTradeX demonstrates how generative AI and retrieval systems can be applied to real operational business problems. It provides a practical foundation for building an AI assistant that helps post-trade teams answer complex procedural and operational questions faster, more consistently, and with better traceability.

---

## License

This project is intended for internal learning and operational experimentation. Please review and adapt the licensing terms before production deployment.
