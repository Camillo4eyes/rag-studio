"""Centralised configuration via Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, resolved from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )
    ollama_model: str = Field(default="llama3", description="Ollama model name")

    # ── Embedder ──────────────────────────────────────────────────────────────
    embedder_provider: Literal["openai", "sentence_transformer"] = Field(
        default="sentence_transformer", description="Embedding provider"
    )
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2", description="SentenceTransformer model name"
    )

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_provider: Literal["chroma", "faiss"] = Field(
        default="chroma", description="Vector store backend"
    )
    chroma_persist_dir: str = Field(
        default="./chroma_data", description="ChromaDB persistence directory"
    )
    chroma_collection: str = Field(
        default="rag_studio", description="ChromaDB collection name"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, description="Target chunk size in tokens/chars")
    chunk_overlap: int = Field(default=64, description="Overlap between chunks")
    chunker_strategy: Literal["fixed", "recursive", "semantic"] = Field(
        default="recursive", description="Chunking strategy"
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, description="Number of documents to retrieve")
    retrieval_score_threshold: float = Field(
        default=0.0, description="Minimum similarity score for retrieval"
    )
    retrieval_method: Literal["similarity", "mmr"] = Field(
        default="similarity", description="Retrieval method"
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    log_level: str = Field(default="info", description="Log level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
