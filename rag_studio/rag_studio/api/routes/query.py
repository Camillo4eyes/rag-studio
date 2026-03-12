"""Query / chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rag_studio.api.models import QueryRequest, QueryResponse, SourceDocument

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse, summary="RAG query")
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Answer a question using the RAG pipeline."""
    pipeline = _get_pipeline(request)

    # Temporarily override retriever settings from request
    pipeline.retriever.top_k = body.top_k
    pipeline.retriever.score_threshold = body.score_threshold
    pipeline.retriever.method = body.method

    try:
        result = pipeline.query(body.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceDocument(
            content=s.content,
            metadata=s.metadata,
            score=s.score,
            doc_id=s.doc_id,
        )
        for s in result.sources
    ]
    return QueryResponse(answer=result.answer, query=result.query, sources=sources)


@router.post("/stream", summary="Streaming RAG query")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Stream the answer to a question using the RAG pipeline."""
    pipeline = _get_pipeline(request)

    pipeline.retriever.top_k = body.top_k
    pipeline.retriever.score_threshold = body.score_threshold
    pipeline.retriever.method = body.method

    def generate():  # type: ignore[return]
        try:
            for token in pipeline.stream_query(body.question):
                yield token
        except Exception as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(generate(), media_type="text/plain")


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline(request: Request):  # type: ignore[return]
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline
