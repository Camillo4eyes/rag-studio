"""ChromaDB vector store implementation."""

from __future__ import annotations

import uuid
from typing import Any

from rag_studio.stores.base import VectorStore


class ChromaStore(VectorStore):
    """Persistent vector store backed by ChromaDB.

    Args:
        collection_name: ChromaDB collection name.
        persist_dir: Directory to persist the database.  Use ``None`` for an
            in-memory (ephemeral) instance.
    """

    def __init__(
        self,
        collection_name: str = "rag_studio",
        persist_dir: str | None = "./chroma_data",
    ) -> None:
        try:
            import chromadb  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'chromadb' to use ChromaStore") from exc

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add(self, items: list[dict[str, Any]]) -> list[str]:
        """Add *items* to the ChromaDB collection."""
        if not items:
            return []

        ids = [str(uuid.uuid4()) for _ in items]
        embeddings = [item["embedding"] for item in items]
        documents = [item["content"] for item in items]
        metadatas = [item.get("metadata", {}) for item in items]

        # ChromaDB requires metadata values to be str | int | float | bool
        cleaned_metadatas = [_sanitise_metadata(m) for m in metadatas]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas,
        )
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the collection for the *top_k* nearest neighbours."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count()) if self.count() > 0 else top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        if self.count() == 0:
            return []

        results = self._collection.query(**kwargs)

        output: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma uses cosine *distance* (0 = identical); convert to similarity
            score = 1.0 - float(dist)
            output.append(
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta or {},
                    "score": score,
                }
            )
        return output

    def delete(self, ids: list[str]) -> None:
        """Delete items from the collection by ID."""
        if ids:
            self._collection.delete(ids=ids)

    def clear(self) -> None:
        """Delete all items from the collection."""
        all_ids = self._collection.get(include=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self._collection.count()


# ── helpers ───────────────────────────────────────────────────────────────────

def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Convert metadata values to ChromaDB-compatible primitives."""
    result: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            result[k] = v
        else:
            result[k] = str(v)
    return result
