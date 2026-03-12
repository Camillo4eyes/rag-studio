"""Embedding generation providers for RAG Studio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    from openai import OpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment,misc]


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: Input strings to embed.

        Returns:
            A list of embedding vectors (one per input string).
        """

    def embed_one(self, text: str) -> list[float]:
        """Convenience method to embed a single text.

        Args:
            text: Input string.

        Returns:
            A single embedding vector.
        """
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced embeddings."""


# ── OpenAI embedder ───────────────────────────────────────────────────────────

class OpenAIEmbedder(Embedder):
    """Embedder backed by the OpenAI Embeddings API.

    Args:
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when *None*.
        model: Embedding model name (default: ``text-embedding-3-small``).
        batch_size: Maximum number of texts per API request.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 512,
    ) -> None:
        if OpenAI is None:
            raise ImportError("Install 'openai' to use OpenAIEmbedder")

        import os
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # probe with a dummy call
            result = self._client.embeddings.create(input=["hello"], model=self.model)
            self._dim = len(result.data[0].embedding)
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches using the OpenAI API."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(input=batch, model=self.model)
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings


# ── SentenceTransformer embedder ──────────────────────────────────────────────

class SentenceTransformerEmbedder(Embedder):
    """Local embedder using the ``sentence-transformers`` library.

    Args:
        model_name: HuggingFace model name (default: ``all-MiniLM-L6-v2``).
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda"``.
            Defaults to auto-detection.
        batch_size: Batch size for inference.
        show_progress_bar: Whether to display a tqdm progress bar.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "Install 'sentence-transformers' to use SentenceTransformerEmbedder"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* locally using sentence-transformers."""
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedder(
    provider: str = "sentence_transformer",
    **kwargs: Any,
) -> Embedder:
    """Return an :class:`Embedder` for the given *provider*.

    Args:
        provider: One of ``"openai"`` or ``"sentence_transformer"``.
        **kwargs: Forwarded to the embedder constructor.

    Raises:
        ValueError: For unknown providers.
    """
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    if provider == "sentence_transformer":
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder provider '{provider}'. Choose from: openai, sentence_transformer")
