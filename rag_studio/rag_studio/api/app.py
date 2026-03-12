"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_studio import __version__
from rag_studio.api.routes import documents, health, query
from rag_studio.config import get_settings
from rag_studio.core.chunker import get_chunker
from rag_studio.core.generator import get_generator
from rag_studio.core.pipeline import RAGPipeline


def _build_pipeline() -> RAGPipeline:
    """Construct a :class:`RAGPipeline` from the current settings."""
    settings = get_settings()

    # Embedder
    if settings.embedder_provider == "openai":
        from rag_studio.core.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )
    else:
        from rag_studio.core.embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(
            model_name=settings.sentence_transformer_model,
        )

    # Vector store
    if settings.vector_store_provider == "faiss":
        from rag_studio.stores.faiss_store import FAISSStore
        store = FAISSStore(dimension=embedder.dimension)
    else:
        from rag_studio.stores.chroma_store import ChromaStore
        store = ChromaStore(
            collection_name=settings.chroma_collection,
            persist_dir=settings.chroma_persist_dir,
        )

    # Generator
    if settings.openai_api_key:
        generator = get_generator(
            "openai",
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
    else:
        generator = get_generator("ollama", model=settings.ollama_model)

    chunker = get_chunker(
        strategy=settings.chunker_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        store=store,
        generator=generator,
        top_k=settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
        retrieval_method=settings.retrieval_method,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise and tear down the RAG pipeline."""
    app.state.pipeline = _build_pipeline()
    yield
    # Clean-up could go here


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Studio API",
        description="A modular Retrieval-Augmented Generation framework",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")

    return app


app = create_app()
