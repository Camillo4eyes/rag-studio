"""Plain text and Markdown document loader."""

from __future__ import annotations

from pathlib import Path

from rag_studio.loaders.base import Document, DocumentLoader


class TextLoader(DocumentLoader):
    """Load plain text or Markdown files as a single :class:`Document`.

    Args:
        path: Path to the text file.
        encoding: File encoding (default: ``utf-8``).
    """

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        """Read the file and return it as a single :class:`Document`.

        Returns:
            A one-element list containing the loaded document.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        content = self.path.read_text(encoding=self.encoding)
        suffix = self.path.suffix.lower().lstrip(".")
        file_type = "markdown" if suffix in {"md", "markdown"} else "text"

        return [
            Document(
                content=content.strip(),
                metadata={
                    "source": str(self.path),
                    "file_type": file_type,
                    "file_name": self.path.name,
                    "size_bytes": self.path.stat().st_size,
                },
                source=str(self.path),
            )
        ]
