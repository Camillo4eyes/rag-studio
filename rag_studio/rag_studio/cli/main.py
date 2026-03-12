"""RAG Studio CLI — built with Typer and Rich."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rag_studio import __version__

app = typer.Typer(
    name="rag-studio",
    help="🧠 RAG Studio — Modular Retrieval-Augmented Generation framework",
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()


# ── Lazy pipeline builder ─────────────────────────────────────────────────────

def _get_pipeline():  # type: ignore[return]
    """Build and return a RAGPipeline from current settings."""
    from rag_studio.api.app import _build_pipeline
    return _build_pipeline()


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into directories"),
    glob_pattern: str = typer.Option("*", "--pattern", "-p", help="File glob pattern for directories"),
) -> None:
    """Ingest documents from a file or directory into the vector store."""
    from rag_studio.loaders.base import DocumentLoader

    pipeline = _get_pipeline()
    paths: list[Path] = []

    if path.is_file():
        paths = [path]
    elif path.is_dir():
        pattern = f"**/{glob_pattern}" if recursive else glob_pattern
        paths = list(path.glob(pattern))
        paths = [p for p in paths if p.is_file()]
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        raise typer.Exit(1)

    if not paths:
        console.print("[yellow]No files found.[/yellow]")
        raise typer.Exit(0)

    total_chunks = 0
    total_docs = 0
    with console.status("[bold green]Ingesting documents…") as status:
        for p in paths:
            try:
                loader = DocumentLoader.from_file(p)
                docs = loader.load()
                n = pipeline.ingest_documents(docs)
                total_chunks += n
                total_docs += 1
                status.update(f"[green]✓[/green] {p.name} ({n} chunks)")
            except ValueError:
                console.print(f"[yellow]Skipping unsupported file: {p.name}[/yellow]")
            except Exception as exc:
                console.print(f"[red]Error processing {p.name}: {exc}[/red]")

    console.print(
        Panel(
            f"✅ Ingested [bold]{total_docs}[/bold] documents → [bold]{total_chunks}[/bold] chunks",
            title="Ingestion complete",
            border_style="green",
        )
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of sources to retrieve"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show source documents"),
) -> None:
    """Ask a question and get a RAG-powered answer."""
    pipeline = _get_pipeline()
    pipeline.retriever.top_k = top_k

    with console.status("[bold blue]Thinking…"):
        result = pipeline.query(question)

    console.print(Panel(result.answer, title="[bold]Answer[/bold]", border_style="blue"))

    if show_sources and result.sources:
        table = Table(title="Sources", show_header=True)
        table.add_column("#", style="dim")
        table.add_column("Score", style="cyan")
        table.add_column("Content", style="white", max_width=80)
        for i, src in enumerate(result.sources, 1):
            table.add_row(str(i), f"{src.score:.3f}", src.content[:200] + "…")
        console.print(table)


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    pipeline = _get_pipeline()
    console.print(Panel("🧠 [bold]RAG Studio Chat[/bold] — type 'exit' to quit", border_style="magenta"))

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

        with console.status("[bold blue]Thinking…"):
            result = pipeline.query(question)

        console.print(f"[bold blue]Assistant:[/bold blue] {result.answer}\n")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload (dev mode)"),
    workers: int = typer.Option(1, help="Number of worker processes"),
) -> None:
    """Start the RAG Studio API server."""
    try:
        import uvicorn  # type: ignore[import]
    except ImportError:
        console.print("[red]Install 'uvicorn' to use the serve command.[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"🚀 Starting RAG Studio API at [bold]http://{host}:{port}[/bold]",
            border_style="green",
        )
    )
    uvicorn.run(
        "rag_studio.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@app.command()
def status() -> None:
    """Show the current system status."""
    pipeline = _get_pipeline()

    table = Table(title="RAG Studio Status", show_header=True)
    table.add_column("Component", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Version", __version__)
    table.add_row("Embedder", type(pipeline.embedder).__name__)
    table.add_row("Vector Store", type(pipeline.store).__name__)
    table.add_row("Generator", type(pipeline.generator).__name__)
    table.add_row("Chunker", type(pipeline.chunker).__name__)

    try:
        doc_count = pipeline.store.count()
        table.add_row("Indexed Chunks", str(doc_count))
    except Exception:
        table.add_row("Indexed Chunks", "N/A")

    console.print(table)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-V", is_eager=True),
) -> None:
    """RAG Studio — Modular RAG framework."""
    if version:
        console.print(f"RAG Studio v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
