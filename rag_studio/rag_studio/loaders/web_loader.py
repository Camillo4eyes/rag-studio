"""Web page loader using requests + BeautifulSoup."""

from __future__ import annotations

try:
    import requests  # type: ignore[import]
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore[assignment,misc]

from rag_studio.loaders.base import Document, DocumentLoader


class WebLoader(DocumentLoader):
    """Load and clean the text content of a web page.

    Args:
        url: The URL to fetch.
        timeout: HTTP request timeout in seconds.
        headers: Optional HTTP headers to send.
        tags_to_extract: HTML tags whose text should be extracted.
            Defaults to common content tags.
    """

    _DEFAULT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "pre", "code"]

    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        tags_to_extract: list[str] | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": (
                "Mozilla/5.0 (compatible; RAGStudio/0.1; +https://github.com/Camillo4eyes/rag-studio)"
            )
        }
        self.tags_to_extract = tags_to_extract or self._DEFAULT_TAGS

    def load(self) -> list[Document]:
        """Fetch the URL and extract clean text.

        Returns:
            A one-element list with the extracted page content.

        Raises:
            ImportError: If ``requests`` or ``beautifulsoup4`` are not installed.
            requests.HTTPError: If the HTTP response indicates an error.
        """
        if requests is None or BeautifulSoup is None:
            raise ImportError(
                "Install 'requests' and 'beautifulsoup4' to use WebLoader"
            )

        response = requests.get(self.url, timeout=self.timeout, headers=self.headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script / style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        texts: list[str] = []
        for tag in soup.find_all(self.tags_to_extract):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)

        content = "\n\n".join(texts)

        return [
            Document(
                content=content,
                metadata={
                    "source": self.url,
                    "title": title,
                    "file_type": "web",
                    "url": self.url,
                },
                source=self.url,
            )
        ]
