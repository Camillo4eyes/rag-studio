"""Source code document loader with language detection."""

from __future__ import annotations

from pathlib import Path

from rag_studio.loaders.base import Document, DocumentLoader


class CodeLoader(DocumentLoader):
    """Load source code files as :class:`~rag_studio.loaders.base.Document` objects.

    Language is auto-detected from the file extension.

    Args:
        path: Path to the source code file.
        encoding: File encoding (default: ``utf-8``).
    """

    SUPPORTED_EXTENSIONS: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".fish": "bash",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".xml": "xml",
        ".lua": "lua",
        ".ex": "elixir",
        ".exs": "elixir",
        ".hs": "haskell",
        ".ml": "ocaml",
    }

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    @property
    def language(self) -> str:
        """Detected programming language based on the file extension."""
        return self.SUPPORTED_EXTENSIONS.get(self.path.suffix.lower(), "unknown")

    def load(self) -> list[Document]:
        """Load the source code file.

        Returns:
            A one-element list containing the code document.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Source file not found: {self.path}")

        content = self.path.read_text(encoding=self.encoding)

        return [
            Document(
                content=content,
                metadata={
                    "source": str(self.path),
                    "file_type": "code",
                    "language": self.language,
                    "file_name": self.path.name,
                    "extension": self.path.suffix.lower(),
                    "size_bytes": self.path.stat().st_size,
                    "line_count": content.count("\n") + 1,
                },
                source=str(self.path),
            )
        ]
