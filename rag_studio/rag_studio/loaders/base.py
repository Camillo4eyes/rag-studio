"""Abstract base classes for document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """A loaded document with its textual content and metadata.

    Attributes:
        content: The extracted text of the document.
        metadata: Arbitrary key-value metadata (source path, page numbers, …).
        source: Human-readable source identifier (file path or URL).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self) -> None:
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source


class DocumentLoader(ABC):
    """Abstract base class for all document loaders.

    Subclasses must implement :meth:`load`, which returns a list of
    :class:`Document` objects extracted from the underlying source.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return a list of :class:`Document` objects.

        Returns:
            Extracted documents.
        """

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "DocumentLoader":
        """Auto-select the right loader based on *path*'s file extension.

        Args:
            path: File system path to load.
            **kwargs: Forwarded to the loader constructor.

        Returns:
            An instantiated loader appropriate for the file type.

        Raises:
            ValueError: If the file extension is not supported.
        """
        from rag_studio.loaders.code_loader import CodeLoader
        from rag_studio.loaders.pdf_loader import PDFLoader
        from rag_studio.loaders.text_loader import TextLoader

        suffix = Path(path).suffix.lower()
        if suffix == ".pdf":
            return PDFLoader(path, **kwargs)
        if suffix in {".txt", ".md", ".rst", ".markdown"}:
            return TextLoader(path, **kwargs)
        if suffix in CodeLoader.SUPPORTED_EXTENSIONS:
            return CodeLoader(path, **kwargs)
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Supported: .pdf, .txt, .md, .rst, and common source code extensions."
        )
