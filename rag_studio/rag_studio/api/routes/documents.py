"""Document management endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import File as FastAPIFile

from rag_studio.api.models import DeleteResponse, DocumentListResponse, DocumentResponse
from rag_studio.loaders.base import DocumentLoader

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory registry: maps document_id → DocumentResponse
# (In a production system this would be a database)
_REGISTRY: dict[str, DocumentResponse] = {}


@router.post("/upload", response_model=DocumentResponse, summary="Upload and index a document")
async def upload_document(
    request: Request,
    file: UploadFile = FastAPIFile(...),
) -> DocumentResponse:
    """Upload a file, chunk it, embed it, and add it to the vector store."""
    pipeline = _get_pipeline(request)

    suffix = Path(file.filename or "upload.txt").suffix.lower()
    allowed = {".pdf", ".txt", ".md", ".py", ".js", ".ts", ".java", ".go", ".rs"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = DocumentLoader.from_file(tmp_path)
        documents = loader.load()
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {exc}") from exc

    chunk_count = pipeline.ingest_documents(documents)
    tmp_path.unlink(missing_ok=True)

    import uuid
    doc_id = str(uuid.uuid4())
    meta = documents[0].metadata if documents else {}
    doc_resp = DocumentResponse(
        id=doc_id,
        source=file.filename or "unknown",
        file_type=meta.get("file_type", suffix.lstrip(".")),
        chunk_count=chunk_count,
        metadata=meta,
    )
    _REGISTRY[doc_id] = doc_resp
    return doc_resp


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents() -> DocumentListResponse:
    """Return the list of all indexed documents."""
    docs = list(_REGISTRY.values())
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(doc_id: str) -> DeleteResponse:
    """Remove a document from the registry.

    Note: This removes the registry entry but does not remove individual chunks
    from the vector store (which would require tracking chunk IDs).
    """
    if doc_id not in _REGISTRY:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    del _REGISTRY[doc_id]
    return DeleteResponse(deleted_ids=[doc_id])


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_pipeline(request: Request):  # type: ignore[return]
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return pipeline
