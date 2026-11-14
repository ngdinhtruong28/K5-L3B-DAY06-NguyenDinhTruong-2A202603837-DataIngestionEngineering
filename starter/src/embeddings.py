"""Real embedding providers using their HTTPS APIs; no mock-vector fallback."""
import json
import math
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class Embedder(Protocol):
    model_name: str

    @property
    def dimensions(self) -> int:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with stored document vectors."""
        ...


class MyEmbedder:
    """OpenAI, OpenRouter, or Gemini embedding API adapter."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("EMBEDDING_MODEL must name a real embedding model")
        self.model_name = model_name.strip()
        self.total_input_tokens = 0
        self.token_usage_complete = True
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower() or "openai"
        configured_dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
        try:
            self._dimensions = int(configured_dimensions) if configured_dimensions else None
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer") from exc
        if self._dimensions is not None and self._dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be a positive integer")

        if self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.base_url = os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
        elif self.provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        else:
            raise ValueError(
                "EMBEDDING_PROVIDER must be one of: openai, openrouter, gemini"
            )
        if not self.api_key:
            raise ValueError(f"API key for embedding provider '{self.provider}' is missing")

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("Embedding dimensions are unknown until a vector is returned")
        return self._dimensions

    def _request(self, payload: dict, url: str, headers: dict) -> dict:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if self.provider == "openai" and exc.code == 403:
                raise RuntimeError(
                    "OpenAI denied this embedding request. Check that the API key's "
                    f"project can use model '{self.model_name}' and the embeddings "
                    f"endpoint. Provider detail: {detail}"
                ) from exc
            raise RuntimeError(
                f"Embedding provider returned HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach embedding provider: {exc.reason}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Embedding provider returned an invalid response")
        return result

    def _record_token_usage(self, response: dict) -> None:
        usage = response.get("usage")
        token_count = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if self.provider == "gemini":
            usage = response.get("usageMetadata")
            token_count = (
                usage.get("promptTokenCount") if isinstance(usage, dict) else None
            )
        if isinstance(token_count, int) and not isinstance(token_count, bool) and token_count >= 0:
            self.total_input_tokens += token_count
        else:
            self.token_usage_complete = False

    def _validate_vectors(self, vectors: list[list[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {expected} inputs"
            )
        checked = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("Embedding provider returned an empty or invalid vector")
            try:
                values = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Embedding provider returned non-numeric vector values") from exc
            if any(not math.isfinite(value) for value in values):
                raise RuntimeError("Embedding provider returned a non-finite vector value")
            if not any(value != 0 for value in values):
                raise RuntimeError("Embedding provider returned a zero vector")
            if self._dimensions is None:
                self._dimensions = len(values)
            if len(values) != self._dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self._dimensions}, got {len(values)}"
                )
            checked.append(values)
        return checked

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts}
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions
        response = self._request(
            payload,
            f"{self.base_url}/embeddings",
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding provider response is missing its data array")
        try:
            data = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Embedding provider returned malformed embedding data") from exc
        self._record_token_usage(response)
        return self._validate_vectors(vectors, len(texts))

    def _embed_gemini(self, texts: list[str], task_type: str) -> list[list[float]]:
        model = f"models/{self.model_name.removeprefix('models/')}"
        payload = {
            "requests": [
                {
                    "model": model,
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    **(
                        {"outputDimensionality": self._dimensions}
                        if self._dimensions is not None
                        else {}
                    ),
                }
                for text in texts
            ]
        }
        url = (
            f"{self.base_url}/{quote(model, safe='/')}:batchEmbedContents?"
            f"{urlencode({'key': self.api_key})}"
        )
        response = self._request(payload, url, {"Content-Type": "application/json"})
        data = response.get("embeddings")
        if not isinstance(data, list):
            raise RuntimeError("Gemini response is missing its embeddings array")
        try:
            vectors = [item["values"] for item in data]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Gemini returned malformed embedding data") from exc
        self._record_token_usage(response)
        return self._validate_vectors(vectors, len(texts))

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self.provider == "gemini":
            return self._embed_gemini(texts, task_type)
        return self._embed_openai(texts)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 64):
            vectors.extend(self._embed_batch(texts[start:start + 64], task_type))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self._embed([question], "RETRIEVAL_QUERY")[0]
