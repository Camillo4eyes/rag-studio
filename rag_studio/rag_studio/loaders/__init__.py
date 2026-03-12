"""Document loaders package."""

from rag_studio.loaders.base import Document, DocumentLoader
from rag_studio.loaders.code_loader import CodeLoader
from rag_studio.loaders.pdf_loader import PDFLoader
from rag_studio.loaders.text_loader import TextLoader
from rag_studio.loaders.web_loader import WebLoader

__all__ = [
    "Document",
    "DocumentLoader",
    "CodeLoader",
    "PDFLoader",
    "TextLoader",
    "WebLoader",
]
