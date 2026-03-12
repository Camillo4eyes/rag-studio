"""Pydantic request / response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Document models ───────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Metadata about an ingested document."""

    id: str
    source: str
    file_type: str = ""
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """List of ingested documents."""

    documents: list[DocumentResponse]
    total: int


class DeleteResponse(BaseModel):
    """Confirmation of a document deletion."""

    deleted_ids: list[str]
    message: str = "Documents deleted successfully"


# ── Query models ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """RAG query request body."""

    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of documents to retrieve")
    score_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    method: str = Field(
        default="similarity",
        description="Retrieval method: 'similarity' or 'mmr'",
    )


class SourceDocument(BaseModel):
    """A retrieved source document included in a query response."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    doc_id: str = ""


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    query: str
    sources: list[SourceDocument] = Field(default_factory=list)


# ── Health models ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    embedder: str = ""
    vector_store: str = ""
    document_count: int = 0
