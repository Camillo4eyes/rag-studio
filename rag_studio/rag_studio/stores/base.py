"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract interface for vector stores.

    Every implementation must support the four core operations:
    :meth:`add`, :meth:`search`, :meth:`delete`, and :meth:`clear`.
    """

    @abstractmethod
    def add(self, items: list[dict[str, Any]]) -> list[str]:
        """Add items to the store.

        Each item dict must contain:
        - ``"content"`` (``str``): the text content.
        - ``"embedding"`` (``list[float]``): the embedding vector.
        - ``"metadata"`` (``dict``, optional): arbitrary metadata.

        Args:
            items: List of item dictionaries.

        Returns:
            List of generated or assigned IDs.
        """

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for the *top_k* most similar items.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (implementation-specific).

        Returns:
            List of result dicts with keys ``"id"``, ``"content"``,
            ``"metadata"``, and ``"score"``.
        """

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete items by their IDs.

        Args:
            ids: IDs of items to remove.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all items from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of items in the store."""
