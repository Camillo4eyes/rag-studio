"""Tests for chunking strategies."""

from __future__ import annotations

import pytest

from rag_studio.core.chunker import (
    Chunk,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)


LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.\n\n"
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum. "
    "Excepteur sint occaecat cupidatat non proident. "
    "Sunt in culpa qui officia deserunt mollit anim id est laborum."
)


# ── FixedSizeChunker ──────────────────────────────────────────────────────────

class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.split("A" * 200)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert len(c.content) <= 50

    def test_overlap_produces_more_chunks(self):
        no_overlap = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        with_overlap = FixedSizeChunker(chunk_size=50, chunk_overlap=25)
        text = "X" * 200
        assert len(with_overlap.split(text)) >= len(no_overlap.split(text))

    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.split("Hello world")
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        assert chunker.split("") == []

    def test_metadata_propagated(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        meta = {"source": "test.txt"}
        chunks = chunker.split("A" * 200, metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "test.txt"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_indices_sequential(self):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.split("A" * 200)
        for i, c in enumerate(chunks):
            assert c.index == i


# ── RecursiveChunker ──────────────────────────────────────────────────────────

class TestRecursiveChunker:
    def test_basic_split(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.split(LOREM)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.split("Hello world.")
        assert len(chunks) == 1

    def test_empty_text(self):
        chunker = RecursiveChunker()
        assert chunker.split("") == []

    def test_metadata_propagated(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        meta = {"source": "file.md"}
        chunks = chunker.split(LOREM, metadata=meta)
        for c in chunks:
            assert c.metadata["source"] == "file.md"

    def test_respects_chunk_size(self):
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=0)
        chunks = chunker.split(LOREM)
        # Most chunks should be at or below chunk_size
        oversized = [c for c in chunks if len(c.content) > 80]
        # Allow a small number of oversized chunks from indivisible words
        assert len(oversized) <= max(1, len(chunks) * 0.2)

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=50, chunk_overlap=60)


# ── SemanticChunker ───────────────────────────────────────────────────────────

class TestSemanticChunker:
    @pytest.fixture
    def stub_embedder(self):
        """Returns embedder that gives high similarity to consecutive sentences."""
        import numpy as np

        def embed(texts: list[str]) -> list[list[float]]:
            # Produce nearly identical vectors to force grouping
            rng = np.random.default_rng(0)
            base = rng.random(8)
            return [(base + rng.random(8) * 0.01).tolist() for _ in texts]

        return embed

    @pytest.fixture
    def low_sim_embedder(self):
        """Returns embedder that gives orthogonal vectors → many chunks."""
        import numpy as np

        def embed(texts: list[str]) -> list[list[float]]:
            vecs = []
            for i in range(len(texts)):
                v = np.zeros(8)
                v[i % 8] = 1.0
                vecs.append(v.tolist())
            return vecs

        return embed

    def test_groups_similar_sentences(self, stub_embedder):
        chunker = SemanticChunker(sentence_embedder=stub_embedder, breakpoint_threshold=0.9, min_chunk_size=1)
        text = "Sentence one. Sentence two. Sentence three."
        chunks = chunker.split(text)
        # With near-identical embeddings, all sentences should be in one chunk
        assert len(chunks) >= 1

    def test_splits_dissimilar_sentences(self, low_sim_embedder):
        chunker = SemanticChunker(
            sentence_embedder=low_sim_embedder,
            breakpoint_threshold=0.9,
            min_chunk_size=1,
        )
        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth."
        chunks = chunker.split(text)
        assert len(chunks) >= 2

    def test_empty_text(self, stub_embedder):
        chunker = SemanticChunker(sentence_embedder=stub_embedder)
        assert chunker.split("") == []


# ── Factory ───────────────────────────────────────────────────────────────────

class TestGetChunker:
    def test_fixed(self):
        c = get_chunker("fixed", chunk_size=200, chunk_overlap=20)
        assert isinstance(c, FixedSizeChunker)

    def test_recursive(self):
        c = get_chunker("recursive", chunk_size=200, chunk_overlap=20)
        assert isinstance(c, RecursiveChunker)

    def test_semantic_requires_embedder(self):
        with pytest.raises(ValueError, match="sentence_embedder"):
            get_chunker("semantic")

    def test_semantic_with_embedder(self):
        c = get_chunker("semantic", sentence_embedder=lambda x: [[0.0] * 8] * len(x))
        assert isinstance(c, SemanticChunker)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunker"):
            get_chunker("unknown_strategy")
