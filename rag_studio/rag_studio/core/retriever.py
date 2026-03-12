"""Retrieval logic: similarity search, MMR, and score filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rag_studio.stores.base import VectorStore


@dataclass
class RetrievedDocument:
    """A document returned by the retriever with its similarity score."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    doc_id: str = ""


class Retriever:
    """Retrieves relevant documents from a vector store.

    Supports:
    - *similarity*: plain top-k nearest neighbour search.
    - *mmr*: Maximal Marginal Relevance for result diversification.

    Args:
        store: The :class:`~rag_studio.stores.base.VectorStore` to search.
        embedder: An object with an ``embed_one`` method used to embed queries.
        top_k: Number of documents to return.
        score_threshold: Minimum similarity score (0–1) to include a result.
        method: ``"similarity"`` or ``"mmr"``.
        mmr_lambda: Trade-off parameter for MMR (closer to 1 → more relevance,
            closer to 0 → more diversity).
    """

    def __init__(
        self,
        store: "VectorStore",
        embedder: Any,
        top_k: int = 5,
        score_threshold: float = 0.0,
        method: str = "similarity",
        mmr_lambda: float = 0.5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.method = method
        self.mmr_lambda = mmr_lambda

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Retrieve documents relevant to *query*.

        Args:
            query: Natural-language query string.

        Returns:
            Ordered list of :class:`RetrievedDocument` objects.
        """
        query_embedding = self.embedder.embed_one(query)
        if self.method == "mmr":
            return self._mmr_retrieve(query_embedding)
        return self._similarity_retrieve(query_embedding)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _similarity_retrieve(self, query_emb: list[float]) -> list[RetrievedDocument]:
        """Standard top-k similarity search."""
        results = self.store.search(query_emb, top_k=self.top_k)
        docs = [
            RetrievedDocument(
                content=r["content"],
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                doc_id=r.get("id", ""),
            )
            for r in results
        ]
        return [d for d in docs if d.score >= self.score_threshold]

    def _mmr_retrieve(self, query_emb: list[float]) -> list[RetrievedDocument]:
        """Maximal Marginal Relevance retrieval.

        Fetches ``top_k * 4`` candidates then greedily selects the subset that
        balances relevance and diversity.
        """
        candidate_k = max(self.top_k * 4, 20)
        results = self.store.search(query_emb, top_k=candidate_k)
        if not results:
            return []

        # Filter by score threshold first
        results = [r for r in results if r.get("score", 0.0) >= self.score_threshold]
        if not results:
            return []

        # We need embeddings for the candidates; store them if available,
        # otherwise fall back to similarity order.
        candidate_embeddings = [r.get("embedding") for r in results]
        if all(e is not None for e in candidate_embeddings):
            selected_indices = self._mmr_select(
                query_emb,
                [e for e in candidate_embeddings if e is not None],
                self.top_k,
            )
            selected = [results[i] for i in selected_indices]
        else:
            # Embeddings not available — fall back to plain similarity order
            selected = results[: self.top_k]

        return [
            RetrievedDocument(
                content=r["content"],
                metadata=r.get("metadata", {}),
                score=r.get("score", 0.0),
                doc_id=r.get("id", ""),
            )
            for r in selected
        ]

    @staticmethod
    def _cosine_sim(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    def _mmr_select(
        self,
        query_emb: list[float],
        candidate_embs: list[list[float]],
        k: int,
    ) -> list[int]:
        """Return indices of *k* candidates selected via MMR."""
        selected: list[int] = []
        remaining = list(range(len(candidate_embs)))

        query_sims = [self._cosine_sim(query_emb, e) for e in candidate_embs]

        while remaining and len(selected) < k:
            if not selected:
                # Pick the most relevant first
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                def mmr_score(i: int) -> float:
                    relevance = query_sims[i]
                    redundancy = max(
                        self._cosine_sim(candidate_embs[i], candidate_embs[j])
                        for j in selected
                    )
                    return self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return selected
