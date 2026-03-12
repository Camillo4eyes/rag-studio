"""Tests for document loaders."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_studio.loaders.base import Document, DocumentLoader
from rag_studio.loaders.code_loader import CodeLoader
from rag_studio.loaders.pdf_loader import PDFLoader
from rag_studio.loaders.text_loader import TextLoader
from rag_studio.loaders.web_loader import WebLoader


# ── TextLoader ────────────────────────────────────────────────────────────────

class TestTextLoader:
    def test_load_plain_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!", encoding="utf-8")
        docs = TextLoader(f).load()
        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"
        assert docs[0].metadata["file_type"] == "text"

    def test_load_markdown(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nParagraph.", encoding="utf-8")
        docs = TextLoader(f).load()
        assert docs[0].metadata["file_type"] == "markdown"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            TextLoader("/nonexistent/path.txt").load()

    def test_metadata_contains_source(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content")
        docs = TextLoader(f).load()
        assert "source" in docs[0].metadata

    def test_source_field(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("content")
        docs = TextLoader(f).load()
        assert docs[0].source == str(f)


# ── CodeLoader ────────────────────────────────────────────────────────────────

class TestCodeLoader:
    def test_load_python(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("print('hello')\n")
        docs = CodeLoader(f).load()
        assert len(docs) == 1
        assert docs[0].metadata["language"] == "python"

    def test_load_javascript(self, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("console.log('hi');")
        docs = CodeLoader(f).load()
        assert docs[0].metadata["language"] == "javascript"

    def test_language_unknown_extension(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("some content")
        loader = CodeLoader(f)
        assert loader.language == "unknown"

    def test_metadata_fields(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("x = 1\n")
        docs = CodeLoader(f).load()
        meta = docs[0].metadata
        assert "line_count" in meta
        assert "size_bytes" in meta
        assert "extension" in meta

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CodeLoader("/no/such/file.py").load()

    def test_supported_extensions_dict(self):
        assert ".py" in CodeLoader.SUPPORTED_EXTENSIONS
        assert ".go" in CodeLoader.SUPPORTED_EXTENSIONS
        assert ".rs" in CodeLoader.SUPPORTED_EXTENSIONS


# ── PDFLoader ─────────────────────────────────────────────────────────────────

class TestPDFLoader:
    def test_load_mocked_pdf(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")  # fake file

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page content here."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("rag_studio.loaders.pdf_loader.PdfReader", return_value=mock_reader):
            docs = PDFLoader(pdf_path).load()

        assert len(docs) == 1
        assert docs[0].content == "Page content here."
        assert docs[0].metadata["page"] == 1

    def test_empty_pages_skipped(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF empty")

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "   "  # whitespace only
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("rag_studio.loaders.pdf_loader.PdfReader", return_value=mock_reader):
            docs = PDFLoader(pdf_path).load()

        assert docs == []

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PDFLoader("/no/such.pdf").load()


# ── WebLoader ─────────────────────────────────────────────────────────────────

class TestWebLoader:
    def test_load_web_page(self):
        html = "<html><head><title>Test Page</title></head><body><p>Hello web!</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_studio.loaders.web_loader.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            docs = WebLoader("https://example.com").load()

        assert len(docs) == 1
        assert "Hello web!" in docs[0].content
        assert docs[0].metadata["title"] == "Test Page"

    def test_url_stored_in_metadata(self):
        html = "<html><body><p>content</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_studio.loaders.web_loader.requests") as mock_req:
            mock_req.get.return_value = mock_resp
            docs = WebLoader("https://test.example.com/page").load()

        assert docs[0].metadata["url"] == "https://test.example.com/page"


# ── DocumentLoader.from_file ──────────────────────────────────────────────────

class TestAutoLoader:
    def test_pdf_auto(self, tmp_path):
        f = tmp_path / "file.pdf"
        f.write_bytes(b"%PDF")
        loader = DocumentLoader.from_file(f)
        assert isinstance(loader, PDFLoader)

    def test_txt_auto(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hi")
        loader = DocumentLoader.from_file(f)
        assert isinstance(loader, TextLoader)

    def test_md_auto(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text("# hi")
        loader = DocumentLoader.from_file(f)
        assert isinstance(loader, TextLoader)

    def test_py_auto(self, tmp_path):
        f = tmp_path / "file.py"
        f.write_text("x=1")
        loader = DocumentLoader.from_file(f)
        assert isinstance(loader, CodeLoader)

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "file.docx"
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentLoader.from_file(f)
