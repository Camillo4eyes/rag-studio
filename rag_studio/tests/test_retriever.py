"""Tests for the Retriever component."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_studio.core.retriever import Retriever, RetrievedDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_docs(store, embedder, texts: list[str]) -> None:
    embeddings = embedder.embed(texts)
    items = [
        {"content": t, "embedding": e, "metadata": {"idx": i}}
        for i, (t, e) in enumerate(zip(texts, embeddings))
    ]
    store.add(items)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRetriever:
    def test_retrieve_returns_documents(self, in_memory_store, mock_embedder):
        texts = ["cats are great", "dogs are loyal", "fish are quiet"]
        _add_docs(in_memory_store, mock_embedder, texts)

        retriever = Retriever(store=in_memory_store, embedder=mock_embedder, top_k=2)
        results = retriever.retrieve("tell me about cats")
        assert len(results) <= 2
        assert all(isinstance(r, RetrievedDocument) for r in results)

    def test_retrieve_empty_store(self, in_memory_store, mock_embedder):
        retriever = Retriever(store=in_memory_store, embedder=mock_embedder, top_k=5)
        results = retriever.retrieve("anything")
        assert results == []

    def test_score_threshold_filters(self, in_memory_store, mock_embedder):
        texts = ["hello world"]
        _add_docs(in_memory_store, mock_embedder, texts)

        retriever = Retriever(
            store=in_memory_store,
            embedder=mock_embedder,
            top_k=5,
            score_threshold=1.1,  # score can't exceed 1.0, so nothing passes
        )
        results = retriever.retrieve("hello")
        assert results == []

    def test_top_k_respected(self, in_memory_store, mock_embedder):
        texts = [f"document number {i}" for i in range(10)]
        _add_docs(in_memory_store, mock_embedder, texts)

        retriever = Retriever(store=in_memory_store, embedder=mock_embedder, top_k=3)
        results = retriever.retrieve("document")
        assert len(results) <= 3

    def test_mmr_method(self, in_memory_store, mock_embedder):
        texts = [f"topic {i}" for i in range(8)]
        _add_docs(in_memory_store, mock_embedder, texts)

        retriever = Retriever(
            store=in_memory_store,
            embedder=mock_embedder,
            top_k=3,
            method="mmr",
        )
        results = retriever.retrieve("topic")
        assert len(results) <= 3

    def test_retrieved_document_fields(self, in_memory_store, mock_embedder):
        texts = ["sample content"]
        _add_docs(in_memory_store, mock_embedder, texts)

        retriever = Retriever(store=in_memory_store, embedder=mock_embedder, top_k=1)
        results = retriever.retrieve("sample")
        assert results
        doc = results[0]
        assert doc.content == "sample content"
        assert isinstance(doc.score, float)
        assert isinstance(doc.metadata, dict)


class TestMMRSelection:
    def test_mmr_select_returns_k_items(self):
        # Test the internal MMR selection method
        retriever = Retriever(
            store=MagicMock(), embedder=MagicMock(), top_k=3, mmr_lambda=0.5
        )
        candidates = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.5, 0.5]]
        query = [1.0, 0.0]
        selected = retriever._mmr_select(query, candidates, k=3)
        assert len(selected) == 3
        assert len(set(selected)) == 3  # no duplicates

    def test_mmr_select_fewer_candidates_than_k(self):
        retriever = Retriever(store=MagicMock(), embedder=MagicMock(), top_k=5)
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        selected = retriever._mmr_select([1.0, 0.0], candidates, k=5)
        assert len(selected) == 2  # can't return more than available
