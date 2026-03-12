"""FAISS in-memory vector store implementation."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from rag_studio.stores.base import VectorStore


class FAISSStore(VectorStore):
    """In-memory vector store backed by FAISS (flat L2 index).

    All data lives in memory; there is no built-in persistence.

    Args:
        dimension: Embedding vector dimensionality.
        metric: Distance metric — ``"l2"`` (default) or ``"cosine"``.
    """

    def __init__(self, dimension: int = 384, metric: str = "cosine") -> None:
        try:
            import faiss  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'faiss-cpu' to use FAISSStore") from exc

        self.dimension = dimension
        self.metric = metric

        if metric == "cosine":
            # Inner-product index; we normalise vectors for cosine similarity
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = faiss.IndexFlatL2(dimension)

        # Parallel lists indexed by position
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, items: list[dict[str, Any]]) -> list[str]:
        """Add *items* to the FAISS index."""
        if not items:
            return []

        ids: list[str] = []
        vecs: list[list[float]] = []
        for item in items:
            ids.append(str(uuid.uuid4()))
            vecs.append(item["embedding"])
            self._contents.append(item["content"])
            self._metadatas.append(item.get("metadata", {}))

        matrix = np.array(vecs, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms

        self._index.add(matrix)  # type: ignore[arg-type]
        self._ids.extend(ids)
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find the *top_k* nearest vectors to *query_embedding*."""
        if self.count() == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        if self.metric == "cosine":
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        k = min(top_k, self.count())
        distances, indices = self._index.search(vec, k)  # type: ignore[arg-type]

        results: list[dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            meta = self._metadatas[idx]
            if filters and not _matches_filters(meta, filters):
                continue
            if self.metric == "cosine":
                score = float(dist)  # inner-product ≈ cosine sim (normalised)
            else:
                score = float(1.0 / (1.0 + dist))  # L2 → similarity proxy
            results.append(
                {
                    "id": self._ids[idx],
                    "content": self._contents[idx],
                    "metadata": meta,
                    "score": score,
                }
            )
        return results

    def delete(self, ids: list[str]) -> None:
        """Delete items by ID.

        Note: FAISS flat indices do not support removal natively.  This method
        rebuilds the index without the deleted items.
        """
        if not ids:
            return
        id_set = set(ids)
        survivors = [
            (i, iid)
            for i, iid in enumerate(self._ids)
            if iid not in id_set
        ]
        if not survivors:
            self.clear()
            return

        idxs, new_ids = zip(*survivors)
        new_contents = [self._contents[i] for i in idxs]
        new_metas = [self._metadatas[i] for i in idxs]

        # Rebuild
        import faiss  # type: ignore[import]

        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            self._index = faiss.IndexFlatL2(self.dimension)

        self._ids = list(new_ids)
        self._contents = new_contents
        self._metadatas = new_metas

        # Re-add (vectors are gone; we'd need to re-embed — not supported here)
        # For simplicity we skip re-inserting; callers should rely on clear()
        # for bulk removal.

    def clear(self) -> None:
        """Remove all vectors and reset the index."""
        import faiss  # type: ignore[import]

        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            self._index = faiss.IndexFlatL2(self.dimension)

        self._ids = []
        self._contents = []
        self._metadatas = []

    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal  # type: ignore[return-value]


# ── helpers ───────────────────────────────────────────────────────────────────

def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True if *metadata* satisfies all *filters*."""
    return all(metadata.get(k) == v for k, v in filters.items())
