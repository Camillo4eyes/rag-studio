"""End-to-end tests for the RAG pipeline."""

from __future__ import annotations

import pytest

from rag_studio.core.chunker import FixedSizeChunker
from rag_studio.core.pipeline import RAGPipeline, RAGResponse
from rag_studio.loaders.base import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline(mock_embedder, mock_generator, in_memory_store):
    return RAGPipeline(
        chunker=FixedSizeChunker(chunk_size=100, chunk_overlap=10),
        embedder=mock_embedder,
        store=in_memory_store,
        generator=mock_generator,
        top_k=3,
        score_threshold=0.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_ingest_text(self, pipeline):
        n = pipeline.ingest_text("Hello world! " * 20)
        assert n >= 1

    def test_ingest_documents(self, pipeline, sample_documents):
        n = pipeline.ingest_documents(sample_documents)
        assert n >= len(sample_documents)  # at least one chunk per doc

    def test_query_returns_response(self, pipeline):
        pipeline.ingest_text("RAG stands for Retrieval-Augmented Generation.")
        response = pipeline.query("What is RAG?")
        assert isinstance(response, RAGResponse)
        assert response.answer == "This is a mock answer."
        assert response.query == "What is RAG?"

    def test_query_with_sources(self, pipeline):
        pipeline.ingest_text("Python is a programming language. " * 5)
        response = pipeline.query("Tell me about Python")
        assert isinstance(response.sources, list)

    def test_stream_query_yields_tokens(self, pipeline):
        pipeline.ingest_text("Streaming is cool!")
        tokens = list(pipeline.stream_query("stream test"))
        assert len(tokens) > 0
        full = "".join(tokens)
        assert len(full) > 0

    def test_clear_empties_store(self, pipeline):
        pipeline.ingest_text("Data to be removed")
        assert pipeline.store.count() > 0
        pipeline.clear()
        assert pipeline.store.count() == 0

    def test_ingest_metadata_attached(self, pipeline):
        meta = {"source": "unit_test", "author": "pytest"}
        pipeline.ingest_text("Content with metadata", metadata=meta)
        assert pipeline.store.count() >= 1

    def test_empty_store_query(self, pipeline):
        """Query against empty store should return a (possibly empty-context) answer."""
        response = pipeline.query("What is 1+1?")
        assert isinstance(response.answer, str)

    def test_ingest_multiple_documents(self, pipeline):
        docs = [
            Document(content=f"Document {i} content", metadata={"id": i}, source=f"doc{i}.txt")
            for i in range(5)
        ]
        total = pipeline.ingest_documents(docs)
        assert total >= 5
        assert pipeline.store.count() >= 5
