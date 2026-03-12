#!/usr/bin/env python
"""RAG Studio — Quick Start Example.

This example shows how to build a minimal RAG pipeline in ~20 lines of code.
It uses a local SentenceTransformer embedder and an in-memory vector store,
so no API keys are required.
"""

from rag_studio.core.chunker import RecursiveChunker
from rag_studio.core.pipeline import RAGPipeline

# 1. Define a simple mock generator (replace with OpenAIGenerator for real use)
from rag_studio.core.generator import Generator
from typing import Iterator


class EchoGenerator(Generator):
    """Toy generator that echoes the context back."""

    def generate(self, prompt: str, **kwargs) -> str:
        lines = prompt.split("\n")
        context_lines = [l for l in lines if l.strip() and not l.startswith("Question")]
        return " | ".join(context_lines[:3]) if context_lines else "No context found."

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        yield self.generate(prompt)


def main() -> None:
    # ── Step 1: Choose components ─────────────────────────────────────────────
    print("🔧 Initialising RAG Studio pipeline…")

    try:
        from rag_studio.core.embedder import SentenceTransformerEmbedder
        from rag_studio.stores.chroma_store import ChromaStore

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        store = ChromaStore(collection_name="quickstart", persist_dir=None)
    except ImportError:
        print("⚠️  Optional dependencies not installed. Using mock components.")
        # Fallback to fully in-memory mocks
        import numpy as np
        from tests.conftest import InMemoryStore, MockEmbedder  # type: ignore

        embedder = MockEmbedder()  # type: ignore
        store = InMemoryStore()  # type: ignore

    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
    generator = EchoGenerator()

    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=3,
    )

    # ── Step 2: Ingest some text ──────────────────────────────────────────────
    corpus = """
    RAG stands for Retrieval-Augmented Generation.
    It is a technique that combines information retrieval with language model generation.
    The idea is to retrieve relevant documents from a knowledge base and use them
    as context for the language model to generate accurate answers.

    Vector stores are databases optimised for storing and searching high-dimensional
    embeddings. ChromaDB and FAISS are popular choices for local development.

    Embeddings are numerical representations of text that capture semantic meaning.
    Similar texts have similar embeddings, enabling similarity search.
    """

    print("\n📚 Ingesting knowledge base…")
    n_chunks = pipeline.ingest_text(corpus.strip(), metadata={"source": "quickstart"})
    print(f"   ✓ Stored {n_chunks} chunks")

    # ── Step 3: Ask questions ─────────────────────────────────────────────────
    questions = [
        "What is RAG?",
        "What are vector stores used for?",
        "How do embeddings work?",
    ]

    for question in questions:
        print(f"\n❓ {question}")
        response = pipeline.query(question)
        print(f"💬 {response.answer}")
        if response.sources:
            print(f"   📎 {len(response.sources)} source(s) retrieved")

    print("\n✅ Quickstart complete!")


if __name__ == "__main__":
    main()
