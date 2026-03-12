"""LLM generation providers for RAG Studio."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Iterator

try:
    from openai import OpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

try:
    import httpx  # type: ignore[import]
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


_DEFAULT_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question based on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""


class Generator(ABC):
    """Abstract base class for text generation providers."""

    def __init__(self, prompt_template: str | None = None) -> None:
        self.prompt_template = prompt_template or _DEFAULT_PROMPT_TEMPLATE

    def build_prompt(self, question: str, context_docs: list[str]) -> str:
        """Format the prompt template with *question* and *context_docs*.

        Args:
            question: The user's question.
            context_docs: Retrieved document contents to use as context.

        Returns:
            A formatted prompt string.
        """
        context = "\n\n".join(context_docs)
        return self.prompt_template.format(context=context, question=question)

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the given *prompt*.

        Args:
            prompt: The full prompt string.
            **kwargs: Provider-specific generation parameters.

        Returns:
            Generated text string.
        """

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the response token-by-token.

        Args:
            prompt: The full prompt string.
            **kwargs: Provider-specific generation parameters.

        Yields:
            Successive text fragments.
        """


# ── OpenAI generator ──────────────────────────────────────────────────────────

class OpenAIGenerator(Generator):
    """Generator backed by the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.
        model: Chat model name (default: ``gpt-4o-mini``).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        prompt_template: Custom prompt template with ``{context}`` and
            ``{question}`` placeholders.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(prompt_template)
        if OpenAI is None:
            raise ImportError("Install 'openai' to use OpenAIGenerator")

        import os
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the OpenAI API and return the full generated text."""
        response = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content or ""

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the OpenAI response token-by-token."""
        stream = self._client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Ollama generator ──────────────────────────────────────────────────────────

class OllamaGenerator(Generator):
    """Generator backed by a local Ollama instance.

    Args:
        model: Ollama model name (default: ``llama3``).
        base_url: Ollama API base URL.
        temperature: Sampling temperature.
        prompt_template: Custom prompt template.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(prompt_template)
        if httpx is None:
            raise ImportError("Install 'httpx' to use OllamaGenerator")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Call the Ollama ``/api/generate`` endpoint."""
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        return str(response.json().get("response", ""))

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream the Ollama response line-by-line."""
        payload = {
            "model": kwargs.get("model", self.model),
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": kwargs.get("temperature", self.temperature)},
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break


# ── Factory ───────────────────────────────────────────────────────────────────

def get_generator(
    provider: str = "openai",
    **kwargs: Any,
) -> Generator:
    """Return a :class:`Generator` for the given *provider*.

    Args:
        provider: One of ``"openai"`` or ``"ollama"``.
        **kwargs: Forwarded to the generator constructor.

    Raises:
        ValueError: For unknown providers.
    """
    if provider == "openai":
        return OpenAIGenerator(**kwargs)
    if provider == "ollama":
        return OllamaGenerator(**kwargs)
    raise ValueError(f"Unknown generator provider '{provider}'. Choose from: openai, ollama")
