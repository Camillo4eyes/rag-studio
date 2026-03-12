"""RAG Pipeline orchestrator — ties every component together."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from rag_studio.core.chunker import Chunk, Chunker
from rag_studio.core.embedder import Embedder
from rag_studio.core.generator import Generator
from rag_studio.core.retriever import RetrievedDocument, Retriever
from rag_studio.loaders.base import Document
from rag_studio.stores.base import VectorStore


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: list[RetrievedDocument] = field(default_factory=list)
    query: str = ""


class RAGPipeline:
    """End-to-end RAG pipeline.

    Orchestrates the full flow:
    ``document → loader → chunker → embedder → store → retriever → generator``

    Args:
        chunker: A :class:`~rag_studio.core.chunker.Chunker` instance.
        embedder: An :class:`~rag_studio.core.embedder.Embedder` instance.
        store: A :class:`~rag_studio.stores.base.VectorStore` instance.
        generator: A :class:`~rag_studio.core.generator.Generator` instance.
        top_k: Number of documents to retrieve.
        score_threshold: Minimum similarity score threshold.
        retrieval_method: ``"similarity"`` or ``"mmr"``.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        top_k: int = 5,
        score_threshold: float = 0.0,
        retrieval_method: str = "similarity",
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.retriever = Retriever(
            store=store,
            embedder=embedder,
            top_k=top_k,
            score_threshold=score_threshold,
            method=retrieval_method,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_documents(self, documents: list[Document]) -> int:
        """Ingest a list of :class:`~rag_studio.loaders.base.Document` objects.

        Each document is split into chunks, embedded, and stored in the
        vector store.

        Args:
            documents: Documents to ingest.

        Returns:
            Total number of chunks stored.
        """
        total = 0
        for doc in documents:
            chunks = self.chunker.split(doc.content, metadata=doc.metadata)
            total += self._store_chunks(chunks)
        return total

    def ingest_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> int:
        """Ingest a raw text string.

        Args:
            text: Plain text to ingest.
            metadata: Optional metadata to attach to every chunk.

        Returns:
            Number of chunks stored.
        """
        chunks = self.chunker.split(text, metadata=metadata)
        return self._store_chunks(chunks)

    def _store_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store *chunks* in the vector store."""
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        items = []
        for chunk, emb in zip(chunks, embeddings):
            items.append(
                {
                    "content": chunk.content,
                    "embedding": emb,
                    "metadata": chunk.metadata,
                }
            )
        self.store.add(items)
        return len(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, **generation_kwargs: Any) -> RAGResponse:
        """Answer *question* using the full RAG pipeline.

        Args:
            question: The natural-language question.
            **generation_kwargs: Passed through to the generator.

        Returns:
            A :class:`RAGResponse` containing the answer and source documents.
        """
        sources = self.retriever.retrieve(question)
        context_docs = [s.content for s in sources]
        prompt = self.generator.build_prompt(question, context_docs)
        answer = self.generator.generate(prompt, **generation_kwargs)
        return RAGResponse(answer=answer, sources=sources, query=question)

    def stream_query(self, question: str, **generation_kwargs: Any) -> Iterator[str]:
        """Stream the answer to *question* token-by-token.

        Args:
            question: The natural-language question.
            **generation_kwargs: Passed through to the generator's stream method.

        Yields:
            Text fragments.
        """
        sources = self.retriever.retrieve(question)
        context_docs = [s.content for s in sources]
        prompt = self.generator.build_prompt(question, context_docs)
        yield from self.generator.stream(prompt, **generation_kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all documents from the vector store."""
        self.store.clear()
