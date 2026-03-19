# 🧠 RAG Studio

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/Camillo4eyes/rag-studio/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **A modular, production-ready Retrieval-Augmented Generation (RAG) framework in Python.**  
> Plug in your documents, choose your LLM, and start chatting — in minutes.

---

## ✨ Features

- 🔌 **Multi-provider support** — OpenAI, Ollama (local), sentence-transformers (free)
- 📄 **Universal document loaders** — PDF, TXT, Markdown, Web pages, Source code
- 🗄️ **Flexible vector stores** — ChromaDB (persistent) + FAISS (in-memory)
- 🔀 **Multiple retrieval strategies** — Similarity search & Maximal Marginal Relevance (MMR)
- ✂️ **Smart chunking** — Fixed-size, Recursive, and Semantic chunking
- 🌐 **REST API** — Full FastAPI app with upload, query, and streaming endpoints
- ⌨️ **CLI** — Intuitive command-line interface with `ingest`, `query`, `chat`, `serve`
- 🧪 **Fully tested** — Pytest suite with mocks (no API keys needed to run tests)
- 🐳 **Docker ready** — Multi-stage Dockerfile + docker-compose

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              RAG Studio Pipeline            │
                        └─────────────────────────────────────────────┘

  ┌──────────┐   load    ┌────────────┐  split  ┌─────────────┐  embed  ┌───────────┐
  │ Document │ ────────► │   Loader   │ ──────► │   Chunker   │ ──────► │  Embedder │
  │  (file,  │          │ PDF / TXT  │         │ Fixed/Recur-│         │ OpenAI /  │
  │  web,… ) │          │ Web / Code │         │ sive/Seman- │         │ ST / local│
  └──────────┘          └────────────┘         │ tic         │         └─────┬─────┘
                                               └─────────────┘               │
                                                                              │ store
                                                                              ▼
                                                                     ┌───────────────┐
                                                                     │  Vector Store │
                                                                     │ ChromaDB/FAISS│
                                                                     └───────┬───────┘
                                                                             │
   ┌──────────┐  answer  ┌───────────┐ prompt  ┌───────────┐  retrieve      │
   │   User   │ ◄─────── │ Generator │ ◄────── │ Retriever │ ◄─────────────┘
   │  Query   │          │ OpenAI /  │         │ Top-K/MMR │
   └──────────┘          │ Ollama    │         └───────────┘
                         └───────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- (Optional) OpenAI API key or Ollama running locally

### Installation

```bash
# Clone the repository
git clone https://github.com/Camillo4eyes/rag-studio.git
cd rag-studio

# Install with pip (editable mode)
pip install -e .

# Or install only core dependencies
pip install -r requirements.txt
```

### First Steps — Python API

```python
from rag_studio.core.chunker import RecursiveChunker
from rag_studio.core.embedder import SentenceTransformerEmbedder
from rag_studio.core.generator import OpenAIGenerator
from rag_studio.core.pipeline import RAGPipeline
from rag_studio.stores.chroma_store import ChromaStore

# 1. Build the pipeline
pipeline = RAGPipeline(
    chunker=RecursiveChunker(chunk_size=512, chunk_overlap=64),
    embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    store=ChromaStore(persist_dir="./chroma_data"),
    generator=OpenAIGenerator(model="gpt-4o-mini"),
)

# 2. Ingest documents
from rag_studio.loaders.pdf_loader import PDFLoader
docs = PDFLoader("my_document.pdf").load()
pipeline.ingest_documents(docs)

# 3. Query
response = pipeline.query("What are the main conclusions?")
print(response.answer)
```

### First Steps — CLI

```bash
# Ingest a folder of documents
rag-studio ingest ./docs/ --recursive

# Ask a question
rag-studio query "What is the main topic?"

# Start an interactive chat
rag-studio chat

# Launch the REST API
rag-studio serve --port 8000

# Check system status
rag-studio status
```

### First Steps — Docker

```bash
# Copy environment template
cp .env.example .env
# (edit .env to set your OPENAI_API_KEY or other settings)

# Start with Docker Compose
docker-compose up -d

# The API is now available at http://localhost:8000
```

---

## 📦 Installation Options

### From source

```bash
git clone https://github.com/Camillo4eyes/rag-studio.git
cd rag-studio
pip install -e ".[dev]"   # includes dev dependencies
```

### Docker

```bash
docker build -t rag-studio:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... rag-studio:latest
```

---

## 📚 Usage Examples

### Load and query a PDF

```python
from rag_studio.loaders.pdf_loader import PDFLoader
from rag_studio.core.pipeline import RAGPipeline
# ... build pipeline as above

docs = PDFLoader("research_paper.pdf").load()
n = pipeline.ingest_documents(docs)
print(f"Indexed {n} chunks")

response = pipeline.query("Summarise the methodology section")
print(response.answer)
for src in response.sources:
    print(f"  - page {src.metadata.get('page')}, score={src.score:.3f}")
```

### Stream a response

```python
for token in pipeline.stream_query("Explain the key findings"):
    print(token, end="", flush=True)
```

### Use Ollama (local, free)

```python
from rag_studio.core.generator import OllamaGenerator

generator = OllamaGenerator(model="llama3", base_url="http://localhost:11434")
pipeline = RAGPipeline(..., generator=generator)
```

### Use MMR retrieval for diverse results

```python
pipeline = RAGPipeline(..., retrieval_method="mmr")
```

---

## 🌐 API Documentation

Start the server: `rag-studio serve` or `uvicorn rag_studio.api.app:app --reload`

Interactive docs available at: **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/documents/upload` | Upload and index a document |
| `GET` | `/api/documents` | List all indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `POST` | `/api/query` | RAG query (returns full answer) |
| `POST` | `/api/query/stream` | Streaming RAG query |

### Upload a document

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@my_document.pdf"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?", "top_k": 5}'
```

### Stream a response

```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the document", "top_k": 5}'
```

---

## ⚙️ Configuration

All settings can be configured via environment variables or a `.env` file:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `` | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDER_PROVIDER` | `sentence_transformer` | `openai` \| `sentence_transformer` |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model |
| `VECTOR_STORE_PROVIDER` | `chroma` | `chroma` \| `faiss` |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB persistence path |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `CHUNKER_STRATEGY` | `recursive` | `fixed` \| `recursive` \| `semantic` |
| `RETRIEVAL_TOP_K` | `5` | Documents to retrieve |
| `RETRIEVAL_METHOD` | `similarity` | `similarity` \| `mmr` |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |

---

## 📄 Supported File Formats

| Format | Loader | Notes |
|--------|--------|-------|
| PDF | `PDFLoader` | Text extraction with pypdf |
| TXT / Markdown | `TextLoader` | Plain text and `.md` files |
| Web pages | `WebLoader` | BeautifulSoup HTML parsing |
| Python | `CodeLoader` | Language-aware metadata |
| JavaScript / TypeScript | `CodeLoader` | Auto-detected |
| Java, Go, Rust, C/C++ | `CodeLoader` | 40+ extensions supported |
| JSON / YAML / TOML | `CodeLoader` | Config files |
| SQL / HTML / CSS | `CodeLoader` | Web & DB files |

---

## 🧪 Running Tests

```bash
# Run all tests (no API keys needed — everything is mocked)
pytest

# With coverage report
pytest --cov=rag_studio --cov-report=html

# Or use Make
make test
make test-cov
```

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint
ruff check rag_studio/ tests/

# Format
ruff format rag_studio/ tests/

# Type check
mypy rag_studio/
```

---

## 📁 Project Structure

```
rag-studio/
├── rag_studio/
│   ├── config.py              # Centralised settings (Pydantic)
│   ├── core/
│   │   ├── chunker.py         # FixedSize, Recursive, Semantic
│   │   ├── embedder.py        # OpenAI, SentenceTransformer
│   │   ├── retriever.py       # Similarity, MMR
│   │   ├── generator.py       # OpenAI, Ollama
│   │   └── pipeline.py        # Orchestrator
│   ├── loaders/
│   │   ├── pdf_loader.py      # PDF files
│   │   ├── text_loader.py     # TXT / Markdown
│   │   ├── web_loader.py      # Web pages
│   │   └── code_loader.py     # Source code
│   ├── stores/
│   │   ├── chroma_store.py    # ChromaDB
│   │   └── faiss_store.py     # FAISS
│   ├── api/
│   │   ├── app.py             # FastAPI application
│   │   ├── models.py          # Pydantic models
│   │   └── routes/            # Endpoint routers
│   └── cli/
│       └── main.py            # Typer CLI
├── tests/                     # Full pytest suite
├── examples/                  # Usage examples
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure all tests pass and add new tests for new functionality.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Built with ❤️ by <a href="https://github.com/Camillo4eyes">Camillo4eyes</a></p>
