"""Tests for generation providers (all mocked — no API keys needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_studio.core.generator import (
    Generator,
    OllamaGenerator,
    OpenAIGenerator,
    get_generator,
)


# ── OpenAIGenerator ───────────────────────────────────────────────────────────

class TestOpenAIGenerator:
    @pytest.fixture
    def mock_openai(self):
        with patch("rag_studio.core.generator.OpenAI") as mock_cls:
            client = MagicMock()
            mock_cls.return_value = client

            # Stub non-streaming response
            choice = MagicMock()
            choice.message.content = "Test answer from OpenAI."
            client.chat.completions.create.return_value = MagicMock(choices=[choice])
            yield client

    def test_generate_returns_string(self, mock_openai):
        gen = OpenAIGenerator(api_key="sk-test")
        result = gen.generate("Some prompt")
        assert result == "Test answer from OpenAI."

    def test_generate_calls_api(self, mock_openai):
        gen = OpenAIGenerator(api_key="sk-test", model="gpt-4o-mini")
        gen.generate("Hello")
        mock_openai.chat.completions.create.assert_called_once()

    def test_build_prompt_includes_context(self, mock_openai):
        gen = OpenAIGenerator(api_key="sk-test")
        prompt = gen.build_prompt("What is RAG?", ["RAG stands for Retrieval-Augmented Generation."])
        assert "RAG" in prompt
        assert "What is RAG?" in prompt

    def test_stream_yields_tokens(self, mock_openai):
        # Setup streaming mock
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="World"))]

        mock_openai.chat.completions.create.return_value = [chunk1, chunk2]

        gen = OpenAIGenerator(api_key="sk-test")
        tokens = list(gen.stream("prompt"))
        assert "Hello " in tokens
        assert "World" in tokens


# ── OllamaGenerator ───────────────────────────────────────────────────────────

class TestOllamaGenerator:
    @pytest.fixture
    def mock_httpx(self):
        with patch("rag_studio.core.generator.httpx") as mock_http:
            response = MagicMock()
            response.json.return_value = {"response": "Ollama answer here."}
            response.raise_for_status = MagicMock()
            mock_http.post.return_value = response
            yield mock_http

    def test_generate_returns_string(self, mock_httpx):
        gen = OllamaGenerator(model="llama3")
        result = gen.generate("A prompt")
        assert result == "Ollama answer here."

    def test_generate_posts_to_correct_url(self, mock_httpx):
        gen = OllamaGenerator(model="llama3", base_url="http://localhost:11434")
        gen.generate("prompt")
        call_url = mock_httpx.post.call_args[0][0]
        assert "generate" in call_url

    def test_stream_yields_tokens(self):
        import json

        lines = [
            json.dumps({"response": "tok1", "done": False}).encode(),
            json.dumps({"response": "tok2", "done": True}).encode(),
        ]

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = [line.decode() for line in lines]

        with patch("rag_studio.core.generator.httpx") as mock_http:
            mock_http.stream.return_value = mock_response
            gen = OllamaGenerator()
            tokens = list(gen.stream("prompt"))

        assert "tok1" in tokens
        assert "tok2" in tokens


# ── Factory ───────────────────────────────────────────────────────────────────

class TestGetGenerator:
    def test_openai_provider(self):
        with patch("rag_studio.core.generator.OpenAI"):
            gen = get_generator("openai", api_key="sk-test")
            assert isinstance(gen, OpenAIGenerator)

    def test_ollama_provider(self):
        gen = get_generator("ollama", model="llama3")
        assert isinstance(gen, OllamaGenerator)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            get_generator("gpt5-turbo-ultra")


# ── Base class ────────────────────────────────────────────────────────────────

class TestGeneratorBase:
    def test_mock_generator_satisfies_contract(self, mock_generator):
        prompt = mock_generator.build_prompt("Q?", ["ctx"])
        assert "Q?" in prompt
        answer = mock_generator.generate(prompt)
        assert isinstance(answer, str)
        tokens = list(mock_generator.stream(prompt))
        assert isinstance(tokens, list)
