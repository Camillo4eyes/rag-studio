"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from rag_studio import __version__
from rag_studio.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(request: Request) -> HealthResponse:
    """Return the current health status of the API server."""
    pipeline = getattr(request.app.state, "pipeline", None)

    doc_count = 0
    embedder_name = "unknown"
    store_name = "unknown"

    if pipeline is not None:
        try:
            doc_count = pipeline.store.count()
        except Exception:
            pass
        embedder_name = type(pipeline.embedder).__name__
        store_name = type(pipeline.store).__name__

    return HealthResponse(
        status="ok",
        version=__version__,
        embedder=embedder_name,
        vector_store=store_name,
        document_count=doc_count,
    )
