"""Shared pytest fixtures for the RAG Studio test suite."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_studio.core.chunker import FixedSizeChunker, RecursiveChunker
from rag_studio.loaders.base import Document
from rag_studio.stores.base import VectorStore


# ── Fixture: sample text ──────────────────────────────────────────────────────

@pytest.fixture
def sample_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is about animals. "
        "Another paragraph starts here.\n\n"
        "In the second paragraph we discuss more topics. "
        "RAG stands for Retrieval-Augmented Generation. "
        "It combines retrieval and generation for better results."
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    return Document(
        content=sample_text,
        metadata={"source": "test.txt", "file_type": "text"},
        source="test.txt",
    )


@pytest.fixture
def sample_documents(sample_text: str) -> list[Document]:
    return [
        Document(content=sample_text, metadata={"source": f"doc{i}.txt"}, source=f"doc{i}.txt")
        for i in range(3)
    ]


# ── Fixture: mock embedder ────────────────────────────────────────────────────

class MockEmbedder:
    """Deterministic mock embedder that returns fixed-size random-ish vectors."""

    DIMENSION = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(seed=42)
        vecs = rng.random((len(texts), self.DIMENSION)).tolist()
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self.DIMENSION


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


# ── Fixture: mock generator ───────────────────────────────────────────────────

class MockGenerator:
    """Mock generator that returns a canned response."""

    def build_prompt(self, question: str, context_docs: list[str]) -> str:
        context = "\n\n".join(context_docs)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def generate(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        return "This is a mock answer."

    def stream(self, prompt: str, **kwargs):  # type: ignore[return]
        for token in ["This ", "is ", "a ", "mock ", "stream."]:
            yield token


@pytest.fixture
def mock_generator() -> MockGenerator:
    return MockGenerator()


# ── Fixture: in-memory vector store ──────────────────────────────────────────

class InMemoryStore(VectorStore):
    """Simple in-memory store for testing (no FAISS/Chroma required)."""

    def __init__(self) -> None:
        self._items: list[dict] = []
        self._ids: list[str] = []

    def add(self, items):
        import uuid
        ids = []
        for item in items:
            doc_id = str(uuid.uuid4())
            self._ids.append(doc_id)
            self._items.append({**item, "id": doc_id})
            ids.append(doc_id)
        return ids

    def search(self, query_embedding, top_k=5, filters=None):
        if not self._items:
            return []
        results = []
        qvec = np.array(query_embedding, dtype=float)
        for item in self._items:
            emb = np.array(item["embedding"], dtype=float)
            norm = np.linalg.norm(qvec) * np.linalg.norm(emb)
            score = float(np.dot(qvec, emb) / norm) if norm > 0 else 0.0
            results.append({
                "id": item["id"],
                "content": item["content"],
                "metadata": item.get("metadata", {}),
                "score": score,
                "embedding": item["embedding"],
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, ids):
        id_set = set(ids)
        self._items = [i for i in self._items if i["id"] not in id_set]
        self._ids = [i for i in self._ids if i not in id_set]

    def clear(self):
        self._items = []
        self._ids = []

    def count(self):
        return len(self._items)


@pytest.fixture
def in_memory_store() -> InMemoryStore:
    return InMemoryStore()


# ── Fixture: chunkers ─────────────────────────────────────────────────────────

@pytest.fixture
def fixed_chunker() -> FixedSizeChunker:
    return FixedSizeChunker(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=200, chunk_overlap=20)
