"""Text chunking strategies for RAG pipelines."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Chunk:
    """A single text chunk with associated metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0

    def __len__(self) -> int:
        return len(self.content)


class Chunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* into a list of :class:`Chunk` objects.

        Args:
            text: The input text to split.
            metadata: Optional metadata to attach to every produced chunk.

        Returns:
            An ordered list of :class:`Chunk` instances.
        """


# ── Fixed-size chunker ────────────────────────────────────────────────────────

class FixedSizeChunker(Chunker):
    """Splits text into fixed-size character windows with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* with a sliding fixed-size window."""
        meta = metadata or {}
        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(content=chunk_text, metadata=dict(meta), index=idx))
                idx += 1
            start += step
        return chunks


# ── Recursive chunker ─────────────────────────────────────────────────────────

class RecursiveChunker(Chunker):
    """Recursively splits text using a hierarchy of separators.

    Tries paragraph → sentence → word level separators until chunks are small
    enough, similar to LangChain's ``RecursiveCharacterTextSplitter``.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between successive chunks in characters.
        separators: Ordered list of separator strings to try.
    """

    _DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self._DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the provided separator hierarchy."""
        if not separators or len(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        good_splits: list[str] = []
        current: list[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s)
            if current_len + s_len + (len(sep) if current else 0) > self.chunk_size:
                if current:
                    good_splits.append(sep.join(current))
                    # keep overlap
                    overlap_len = 0
                    while current and overlap_len < self.chunk_overlap:
                        overlap_len += len(current[-1])
                        if overlap_len <= self.chunk_overlap:
                            current = current[-1:]
                            break
                        else:
                            current = []
                    else:
                        current = []
                    current_len = sum(len(c) for c in current)
                # If a single split is still too large, recurse
                if s_len > self.chunk_size:
                    sub = self._split_text(s, separators[1:])
                    good_splits.extend(sub[:-1])
                    current = [sub[-1]] if sub else []
                    current_len = len(current[0]) if current else 0
                else:
                    current = [s]
                    current_len = s_len
            else:
                current.append(s)
                current_len += s_len + (len(sep) if len(current) > 1 else 0)

        if current:
            good_splits.append(sep.join(current))

        return good_splits

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* recursively using the separator hierarchy."""
        meta = metadata or {}
        raw = self._split_text(text, self.separators)
        chunks = []
        for idx, piece in enumerate(raw):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(content=piece, metadata=dict(meta), index=idx))
        return chunks


# ── Semantic chunker ──────────────────────────────────────────────────────────

class SemanticChunker(Chunker):
    """Groups sentences into chunks based on semantic similarity.

    Sentences whose embedding is sufficiently similar to the current chunk's
    centroid are merged; a new chunk is started when similarity drops below
    *breakpoint_threshold*.

    Args:
        sentence_embedder: A callable ``(list[str]) -> list[list[float]]`` that
            produces embeddings for a list of sentences.
        breakpoint_threshold: Cosine-similarity threshold below which a new
            chunk is started.
        min_chunk_size: Minimum number of characters for a chunk to be kept.
    """

    def __init__(
        self,
        sentence_embedder: Any,
        breakpoint_threshold: float = 0.7,
        min_chunk_size: int = 50,
    ) -> None:
        self.sentence_embedder = sentence_embedder
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Naïve sentence splitter using punctuation."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split *text* by grouping semantically similar consecutive sentences."""
        meta = metadata or {}
        sentences = self._sentence_split(text)
        if not sentences:
            return []

        embeddings = self.sentence_embedder(sentences)

        chunks: list[Chunk] = []
        current_sentences: list[str] = [sentences[0]]
        current_emb: list[float] = embeddings[0]
        chunk_idx = 0

        for sent, emb in zip(sentences[1:], embeddings[1:]):
            sim = self._cosine_sim(current_emb, emb)
            if sim >= self.breakpoint_threshold:
                # Update centroid as running mean
                n = len(current_sentences)
                current_emb = [(current_emb[i] * n + emb[i]) / (n + 1) for i in range(len(emb))]
                current_sentences.append(sent)
            else:
                content = " ".join(current_sentences)
                if len(content) >= self.min_chunk_size:
                    chunks.append(Chunk(content=content, metadata=dict(meta), index=chunk_idx))
                    chunk_idx += 1
                current_sentences = [sent]
                current_emb = emb

        # flush remaining
        content = " ".join(current_sentences)
        if len(content) >= self.min_chunk_size:
            chunks.append(Chunk(content=content, metadata=dict(meta), index=chunk_idx))

        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any,
) -> Chunker:
    """Return a :class:`Chunker` instance for the given *strategy*.

    Args:
        strategy: One of ``"fixed"``, ``"recursive"``, or ``"semantic"``.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        **kwargs: Extra keyword arguments forwarded to the chunker constructor.

    Raises:
        ValueError: If an unknown strategy is provided.
    """
    strategies: dict[str, type[Chunker]] = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
    }
    if strategy in strategies:
        return strategies[strategy](
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs
        )
    if strategy == "semantic":
        embedder = kwargs.pop("sentence_embedder", None)
        if embedder is None:
            raise ValueError("SemanticChunker requires a 'sentence_embedder' callable")
        return SemanticChunker(sentence_embedder=embedder, **kwargs)
    raise ValueError(f"Unknown chunker strategy '{strategy}'. Choose from: fixed, recursive, semantic")
