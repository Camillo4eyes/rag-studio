"""PDF document loader using pypdf."""

from __future__ import annotations

from pathlib import Path

try:
    from pypdf import PdfReader  # type: ignore[import]
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment,misc]

from rag_studio.loaders.base import Document, DocumentLoader


class PDFLoader(DocumentLoader):
    """Load text from a PDF file, one :class:`~rag_studio.loaders.base.Document`
    per page.

    Args:
        path: Path to the PDF file.
        extract_images: If ``True``, attempt to extract text from images using
            OCR (requires additional dependencies).
    """

    def __init__(self, path: str | Path, extract_images: bool = False) -> None:
        self.path = Path(path)
        self.extract_images = extract_images

    def load(self) -> list[Document]:
        """Extract text from every page of the PDF.

        Returns:
            A list of :class:`Document` objects, one per non-empty page.

        Raises:
            FileNotFoundError: If the file does not exist.
            ImportError: If ``pypdf`` is not installed.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.path}")

        if PdfReader is None:
            raise ImportError("Install 'pypdf' to use PDFLoader")

        reader = PdfReader(str(self.path))
        documents: list[Document] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            documents.append(
                Document(
                    content=text,
                    metadata={
                        "source": str(self.path),
                        "page": page_num + 1,
                        "total_pages": len(reader.pages),
                        "file_type": "pdf",
                    },
                    source=str(self.path),
                )
            )

        return documents
