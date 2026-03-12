"""Tests for vector store implementations."""

from __future__ import annotations

import numpy as np
import pytest

from rag_studio.stores.base import VectorStore
from tests.conftest import InMemoryStore


DIM = 16


def _random_items(n: int, dim: int = DIM) -> list[dict]:
    rng = np.random.default_rng(0)
    return [
        {
            "content": f"document {i}",
            "embedding": rng.random(dim).tolist(),
            "metadata": {"idx": i},
        }
        for i in range(n)
    ]


# ── InMemoryStore (used as reference in other tests) ─────────────────────────

class TestInMemoryStore:
    def test_add_and_count(self, in_memory_store):
        items = _random_items(5)
        ids = in_memory_store.add(items)
        assert len(ids) == 5
        assert in_memory_store.count() == 5

    def test_search_returns_results(self, in_memory_store):
        items = _random_items(10)
        in_memory_store.add(items)
        query = np.random.default_rng(99).random(DIM).tolist()
        results = in_memory_store.search(query, top_k=3)
        assert len(results) <= 3
        for r in results:
            assert "content" in r
            assert "score" in r

    def test_search_empty_store(self, in_memory_store):
        query = [0.0] * DIM
        results = in_memory_store.search(query, top_k=5)
        assert results == []

    def test_delete(self, in_memory_store):
        ids = in_memory_store.add(_random_items(3))
        in_memory_store.delete([ids[0]])
        assert in_memory_store.count() == 2

    def test_clear(self, in_memory_store):
        in_memory_store.add(_random_items(5))
        in_memory_store.clear()
        assert in_memory_store.count() == 0

    def test_add_returns_unique_ids(self, in_memory_store):
        ids = in_memory_store.add(_random_items(5))
        assert len(set(ids)) == 5


# ── ChromaStore ───────────────────────────────────────────────────────────────

class TestChromaStore:
    @pytest.fixture
    def store(self, tmp_path):
        """Create an ephemeral ChromaDB store (no disk I/O), with unique collection per test."""
        pytest.importorskip("chromadb", reason="chromadb not installed")
        import uuid
        from rag_studio.stores.chroma_store import ChromaStore
        return ChromaStore(collection_name=f"test_{uuid.uuid4().hex}", persist_dir=None)

    def test_add_and_count(self, store):
        store.add(_random_items(3, DIM))
        assert store.count() == 3

    def test_search(self, store):
        store.add(_random_items(5, DIM))
        query = np.random.default_rng(1).random(DIM).tolist()
        results = store.search(query, top_k=3)
        assert len(results) <= 3

    def test_clear(self, store):
        store.add(_random_items(4, DIM))
        store.clear()
        assert store.count() == 0

    def test_delete(self, store):
        ids = store.add(_random_items(3, DIM))
        store.delete([ids[0]])
        assert store.count() == 2

    def test_search_empty(self, store):
        query = [0.0] * DIM
        results = store.search(query, top_k=5)
        assert results == []

    def test_score_in_results(self, store):
        store.add(_random_items(3, DIM))
        query = np.random.default_rng(2).random(DIM).tolist()
        results = store.search(query, top_k=2)
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.1  # cosine can be slightly > 1 due to floats


# ── FAISSStore ────────────────────────────────────────────────────────────────

class TestFAISSStore:
    @pytest.fixture
    def store(self):
        pytest.importorskip("faiss", reason="faiss-cpu not installed")
        from rag_studio.stores.faiss_store import FAISSStore
        return FAISSStore(dimension=DIM, metric="cosine")

    def test_add_and_count(self, store):
        store.add(_random_items(5, DIM))
        assert store.count() == 5

    def test_search_returns_results(self, store):
        store.add(_random_items(10, DIM))
        query = np.random.default_rng(7).random(DIM).tolist()
        results = store.search(query, top_k=4)
        assert len(results) <= 4

    def test_search_empty(self, store):
        query = [0.0] * DIM
        results = store.search(query, top_k=3)
        assert results == []

    def test_clear(self, store):
        store.add(_random_items(5, DIM))
        store.clear()
        assert store.count() == 0

    def test_l2_metric(self):
        pytest.importorskip("faiss", reason="faiss-cpu not installed")
        from rag_studio.stores.faiss_store import FAISSStore
        s = FAISSStore(dimension=DIM, metric="l2")
        s.add(_random_items(3, DIM))
        query = np.random.default_rng(3).random(DIM).tolist()
        results = s.search(query, top_k=2)
        assert len(results) <= 2
