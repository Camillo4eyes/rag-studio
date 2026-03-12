"""Tests for the FastAPI REST endpoints."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rag_studio.core.chunker import FixedSizeChunker
from rag_studio.core.pipeline import RAGPipeline, RAGResponse
from rag_studio.core.retriever import RetrievedDocument


# ── App setup ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline(mock_embedder, mock_generator, in_memory_store):
    pipeline = RAGPipeline(
        chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=20),
        embedder=mock_embedder,
        store=in_memory_store,
        generator=mock_generator,
        top_k=3,
    )
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a TestClient with a pre-configured mock pipeline, bypassing lifespan."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from rag_studio import __version__
    from rag_studio.api.routes import documents, health, query

    # Reset the document registry before each test
    documents._REGISTRY.clear()

    @asynccontextmanager
    async def test_lifespan(app):
        app.state.pipeline = mock_pipeline
        yield

    app = FastAPI(
        title="RAG Studio API (Test)",
        version=__version__,
        lifespan=test_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")

    with TestClient(app) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_has_embedder_field(self, client):
        response = client.get("/api/health")
        assert "embedder" in response.json()


# ── Documents ─────────────────────────────────────────────────────────────────

class TestDocumentEndpoints:
    def test_list_documents_empty(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_text_document(self, client):
        content = b"This is a test document for uploading."
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["source"] == "test.txt"

    def test_upload_unsupported_type(self, client):
        response = client.post(
            "/api/documents/upload",
            files={"file": ("file.docx", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_upload_then_list(self, client):
        content = b"Some content for listing."
        client.post(
            "/api/documents/upload",
            files={"file": ("list_test.txt", io.BytesIO(content), "text/plain")},
        )
        response = client.get("/api/documents")
        assert response.json()["total"] >= 1

    def test_delete_nonexistent_document(self, client):
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404

    def test_upload_and_delete(self, client):
        content = b"Delete me!"
        upload_resp = client.post(
            "/api/documents/upload",
            files={"file": ("todelete.txt", io.BytesIO(content), "text/plain")},
        )
        doc_id = upload_resp.json()["id"]
        del_resp = client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert doc_id in del_resp.json()["deleted_ids"]


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQueryEndpoints:
    def test_query_returns_answer(self, client, mock_pipeline):
        # Add some data first
        mock_pipeline.ingest_text("RAG is awesome for knowledge retrieval.")

        response = client.post(
            "/api/query",
            json={"question": "What is RAG?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "query" in data
        assert "sources" in data

    def test_query_empty_question_rejected(self, client):
        response = client.post("/api/query", json={"question": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_stream_query(self, client, mock_pipeline):
        mock_pipeline.ingest_text("Streaming is great.")
        response = client.post(
            "/api/query/stream",
            json={"question": "Tell me about streaming"},
        )
        assert response.status_code == 200
        # Content should be streamed as text
        assert len(response.text) > 0
