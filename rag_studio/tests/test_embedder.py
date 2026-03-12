"""Tests for embedding providers (all mocked — no API keys needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_studio.core.embedder import (
    Embedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


# ── OpenAIEmbedder ────────────────────────────────────────────────────────────

class TestOpenAIEmbedder:
    @pytest.fixture
    def mock_openai_client(self):
        """Patch the OpenAI client used by OpenAIEmbedder."""
        with patch("rag_studio.core.embedder.OpenAI") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client

            def fake_create(input, model):
                data = [MagicMock(embedding=[0.1] * 8) for _ in input]
                return MagicMock(data=data)

            client.embeddings.create.side_effect = fake_create
            yield client

    def test_embed_returns_correct_shape(self, mock_openai_client):
        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert all(len(v) == 8 for v in result)

    def test_embed_empty_list(self, mock_openai_client):
        embedder = OpenAIEmbedder(api_key="sk-test")
        assert embedder.embed([]) == []

    def test_embed_one(self, mock_openai_client):
        embedder = OpenAIEmbedder(api_key="sk-test")
        result = embedder.embed_one("test")
        assert isinstance(result, list)
        assert len(result) == 8

    def test_batching(self, mock_openai_client):
        """Verify that large input is split into batches."""
        embedder = OpenAIEmbedder(api_key="sk-test", batch_size=3)
        texts = [f"text {i}" for i in range(10)]
        results = embedder.embed(texts)
        assert len(results) == 10
        # Should have been called in ceil(10/3)=4 batches
        assert mock_openai_client.embeddings.create.call_count == 4


# ── SentenceTransformerEmbedder ───────────────────────────────────────────────

class TestSentenceTransformerEmbedder:
    @pytest.fixture
    def mock_st(self):
        """Patch sentence_transformers.SentenceTransformer."""
        import numpy as np

        with patch("rag_studio.core.embedder.SentenceTransformer") as mock_cls:
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = 16

            def fake_encode(texts, **kwargs):
                return np.ones((len(texts), 16), dtype=np.float32)

            model.encode.side_effect = fake_encode
            mock_cls.return_value = model
            yield model

    def test_embed_shape(self, mock_st):
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        result = embedder.embed(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 16 for v in result)

    def test_embed_empty(self, mock_st):
        embedder = SentenceTransformerEmbedder()
        assert embedder.embed([]) == []

    def test_dimension(self, mock_st):
        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 16


# ── Shared Embedder contract ──────────────────────────────────────────────────

class TestEmbedderContract:
    """Verify that the mock embedder from conftest satisfies the interface."""

    def test_embed_returns_list_of_lists(self, mock_embedder):
        result = mock_embedder.embed(["hello", "world"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_embed_one_returns_list(self, mock_embedder):
        result = mock_embedder.embed_one("test")
        assert isinstance(result, list)

    def test_embed_length_matches_input(self, mock_embedder):
        texts = ["a", "b", "c", "d"]
        result = mock_embedder.embed(texts)
        assert len(result) == len(texts)


# ── Factory ───────────────────────────────────────────────────────────────────

class TestGetEmbedder:
    def test_openai_provider(self):
        with patch("rag_studio.core.embedder.OpenAI"):
            embedder = get_embedder("openai", api_key="sk-test")
            assert isinstance(embedder, OpenAIEmbedder)

    def test_sentence_transformer_provider(self):
        with patch("rag_studio.core.embedder.SentenceTransformer") as m:
            m.return_value = MagicMock(get_sentence_embedding_dimension=lambda: 16)
            embedder = get_embedder("sentence_transformer")
            assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder"):
            get_embedder("unknown")
