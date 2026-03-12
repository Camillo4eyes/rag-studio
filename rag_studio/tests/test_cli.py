"""Tests for the CLI commands."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rag_studio.cli.main import app


runner = CliRunner()


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.store.count.return_value = 42
    pipeline.query.return_value = MagicMock(
        answer="CLI mock answer", sources=[], query="test"
    )
    pipeline.ingest_documents.return_value = 5
    pipeline.ingest_text.return_value = 3
    pipeline.stream_query.return_value = iter(["Hello", " world"])
    type(pipeline.embedder).__name__ = "MockEmbedder"
    type(pipeline.store).__name__ = "InMemoryStore"
    type(pipeline.generator).__name__ = "MockGenerator"
    type(pipeline.chunker).__name__ = "FixedSizeChunker"
    pipeline.retriever.top_k = 5
    return pipeline


# ── Version ───────────────────────────────────────────────────────────────────

class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "RAG Studio" in result.output


# ── Status ────────────────────────────────────────────────────────────────────

class TestStatus:
    def test_status_command(self, mock_pipeline):
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Version" in result.output or "RAG" in result.output


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQueryCommand:
    def test_query_basic(self, mock_pipeline):
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["query", "What is RAG?"])
        assert result.exit_code == 0
        assert "CLI mock answer" in result.output

    def test_query_with_sources(self, mock_pipeline):
        mock_pipeline.query.return_value = MagicMock(
            answer="Answer",
            query="Q?",
            sources=[
                MagicMock(score=0.9, content="source content"),
            ],
        )
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["query", "Q?", "--sources"])
        assert result.exit_code == 0


# ── Ingest ────────────────────────────────────────────────────────────────────

class TestIngestCommand:
    def test_ingest_file(self, mock_pipeline, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello from ingest test!")
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ingest", str(f)])
        assert result.exit_code == 0

    def test_ingest_directory(self, mock_pipeline, tmp_path):
        for i in range(3):
            (tmp_path / f"doc{i}.txt").write_text(f"Document {i} content")
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--pattern", "*.txt"])
        assert result.exit_code == 0

    def test_ingest_nonexistent_path(self, mock_pipeline):
        with patch("rag_studio.cli.main._get_pipeline", return_value=mock_pipeline):
            result = runner.invoke(app, ["ingest", "/no/such/path"])
        assert result.exit_code != 0


# ── Help ──────────────────────────────────────────────────────────────────────

class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output.lower() or "RAG" in result.output

    def test_query_help(self):
        result = runner.invoke(app, ["query", "--help"])
        assert result.exit_code == 0
