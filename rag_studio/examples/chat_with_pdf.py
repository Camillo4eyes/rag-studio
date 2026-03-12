#!/usr/bin/env python
"""RAG Studio — Chat with a PDF Example.

Usage:
    python examples/chat_with_pdf.py path/to/document.pdf

Requires:
    - pypdf (for PDF loading)
    - sentence-transformers (for local embeddings)
    - chromadb (for vector storage)
    - An LLM: either set OPENAI_API_KEY or have Ollama running locally.
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def build_pipeline(pdf_path: Path):
    """Build a RAGPipeline configured for chatting with a PDF."""
    from rag_studio.core.chunker import RecursiveChunker
    from rag_studio.core.pipeline import RAGPipeline
    from rag_studio.core.embedder import SentenceTransformerEmbedder
    from rag_studio.stores.chroma_store import ChromaStore
    import os

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    store = ChromaStore(
        collection_name=f"pdf_{pdf_path.stem[:20]}",
        persist_dir=None,  # ephemeral — change to a path for persistence
    )

    # Choose generator based on available credentials
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        from rag_studio.core.generator import OpenAIGenerator
        generator = OpenAIGenerator(api_key=openai_key, model="gpt-4o-mini")
        console.print("🤖 Using [bold green]OpenAI[/bold green] as LLM")
    else:
        from rag_studio.core.generator import OllamaGenerator
        generator = OllamaGenerator(model="llama3")
        console.print("🦙 Using [bold blue]Ollama (llama3)[/bold blue] as LLM")

    return RAGPipeline(
        chunker=RecursiveChunker(chunk_size=512, chunk_overlap=64),
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=5,
    )


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage: python chat_with_pdf.py <path/to/file.pdf>[/red]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        console.print(f"[red]File not found: {pdf_path}[/red]")
        sys.exit(1)

    console.print(Panel(f"📄 Loading [bold]{pdf_path.name}[/bold]", border_style="cyan"))

    # Load PDF
    from rag_studio.loaders.pdf_loader import PDFLoader
    with console.status("Parsing PDF…"):
        loader = PDFLoader(pdf_path)
        documents = loader.load()

    console.print(f"   ✓ Loaded {len(documents)} page(s)")

    # Build pipeline and ingest
    pipeline = build_pipeline(pdf_path)

    with console.status("Embedding and indexing…"):
        n_chunks = pipeline.ingest_documents(documents)

    console.print(f"   ✓ Indexed {n_chunks} chunks\n")
    console.print(Panel("💬 [bold]Chat with your PDF[/bold] — type 'exit' to quit", border_style="magenta"))

    # Chat loop
    while True:
        try:
            question = console.input("[bold magenta]You:[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if question.lower() in {"exit", "quit", "q"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if not question:
            continue

        with console.status("[bold blue]Searching & generating…"):
            response = pipeline.query(question)

        console.print(f"[bold blue]Assistant:[/bold blue] {response.answer}")

        if response.sources:
            console.print(f"   [dim]📎 {len(response.sources)} source chunk(s) used[/dim]\n")


if __name__ == "__main__":
    main()
