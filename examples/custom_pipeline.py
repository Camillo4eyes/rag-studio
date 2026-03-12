#!/usr/bin/env python
"""RAG Studio — Custom Pipeline Example.

Demonstrates how to compose a fully customised RAG pipeline:
- SemanticChunker with a custom embedder
- FAISS vector store
- Custom prompt template
- MMR retrieval for diverse results
"""

import os
from typing import Iterator

from rag_studio.core.chunker import RecursiveChunker, SemanticChunker
from rag_studio.core.generator import Generator
from rag_studio.core.pipeline import RAGPipeline


# ── Custom generator with a domain-specific prompt ────────────────────────────

CUSTOM_PROMPT = """\
You are an expert technical assistant specialised in software engineering.
Use ONLY the provided context to answer the question. If the answer is not in the
context, respond with: "The documentation does not cover this topic."

## Context
{context}

## Question
{question}

## Answer (be concise and precise)"""


class CustomMockGenerator(Generator):
    """Demo generator — replace with OpenAIGenerator or OllamaGenerator."""

    def __init__(self) -> None:
        super().__init__(prompt_template=CUSTOM_PROMPT)

    def generate(self, prompt: str, **kwargs) -> str:
        # Extract context from the prompt
        if "Context" in prompt:
            ctx_start = prompt.index("## Context") + len("## Context\n")
            ctx_end = prompt.index("## Question")
            context = prompt[ctx_start:ctx_end].strip()
            return f"[Custom RAG] Based on the docs: {context[:200]}…"
        return "No relevant context found."

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        full = self.generate(prompt)
        for word in full.split():
            yield word + " "


def main() -> None:
    print("🔧 Building custom RAG pipeline…\n")

    # ── Embedder ──────────────────────────────────────────────────────────────
    try:
        from rag_studio.core.embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        print("✓ Embedder: SentenceTransformer (all-MiniLM-L6-v2)")
    except Exception:
        # Fallback mock
        import numpy as np
        from tests.conftest import MockEmbedder  # type: ignore
        embedder = MockEmbedder()  # type: ignore
        print("✓ Embedder: Mock (no sentence-transformers installed)")

    # ── Vector Store — FAISS ──────────────────────────────────────────────────
    try:
        from rag_studio.stores.faiss_store import FAISSStore
        store = FAISSStore(dimension=embedder.dimension, metric="cosine")
        print(f"✓ Vector Store: FAISS (dim={embedder.dimension}, metric=cosine)")
    except Exception:
        from tests.conftest import InMemoryStore  # type: ignore
        store = InMemoryStore()  # type: ignore
        print("✓ Vector Store: InMemory (no faiss-cpu installed)")

    # ── Chunker ───────────────────────────────────────────────────────────────
    # Use SemanticChunker if embedder is available, otherwise fall back
    try:
        chunker = SemanticChunker(
            sentence_embedder=embedder.embed,
            breakpoint_threshold=0.75,
            min_chunk_size=100,
        )
        print("✓ Chunker: SemanticChunker (threshold=0.75)")
    except Exception:
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
        print("✓ Chunker: RecursiveChunker (fallback)")

    # ── Generator ─────────────────────────────────────────────────────────────
    generator = CustomMockGenerator()
    print("✓ Generator: CustomMockGenerator with domain-specific prompt\n")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=3,
        score_threshold=0.0,
        retrieval_method="mmr",  # Use MMR for diverse results
    )

    # ── Ingest documents ──────────────────────────────────────────────────────
    docs = [
        "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python.",
        "Pydantic is a data validation library that uses Python type annotations.",
        "ChromaDB is an open-source embedding database that makes it easy to build LLM applications.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search.",
        "Sentence transformers provide an easy method to compute dense vector representations for sentences.",
        "RAG (Retrieval-Augmented Generation) improves LLM outputs by retrieving relevant context.",
    ]

    print("📚 Ingesting technical documentation…")
    total = 0
    for i, text in enumerate(docs):
        n = pipeline.ingest_text(text, metadata={"doc_id": i, "source": "tech_docs"})
        total += n
    print(f"   ✓ {total} chunk(s) indexed\n")

    # ── Query ─────────────────────────────────────────────────────────────────
    questions = [
        "What is FastAPI used for?",
        "How does FAISS work?",
        "Tell me about RAG.",
    ]

    for q in questions:
        print(f"❓ {q}")
        response = pipeline.query(q)
        print(f"💬 {response.answer[:200]}")
        print(f"   📎 Retrieved {len(response.sources)} source(s)\n")

    # ── Streaming example ─────────────────────────────────────────────────────
    print("🌊 Streaming response example:")
    print("   Q: What is Pydantic?")
    print("   A: ", end="", flush=True)
    for token in pipeline.stream_query("What is Pydantic?"):
        print(token, end="", flush=True)
    print("\n")

    print("✅ Custom pipeline demo complete!")


if __name__ == "__main__":
    main()
