"""Thin client for a local Ollama server (chat + embeddings). No API key needed."""
import requests


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


class OllamaClient:
    def __init__(self, host: str, chat_model: str, embed_model: str,
                 num_ctx: int = 8192, temperature: float = 0.3, timeout: int = 300):
        self.host = host.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    # ---- status -------------------------------------------------------
    def models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except (requests.RequestException, ValueError) as e:
            raise OllamaError(f"Ollama is not reachable at {self.host}") from e

    def is_up(self) -> bool:
        try:
            self.models()
            return True
        except OllamaError:
            return False

    def has_model(self, name: str) -> bool:
        try:
            installed = self.models()
        except OllamaError:
            return False
        wanted = name if ":" in name else f"{name}:latest"
        return any(m == name or m == wanted for m in installed)

    # ---- calls --------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise OllamaError(f"Ollama is not reachable at {self.host}") from e
        if r.status_code != 200:
            try:
                detail = r.json().get("error", r.text)
            except ValueError:
                detail = r.text
            raise OllamaError(f"Ollama error ({r.status_code}): {detail}")
        return r.json()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns the assistant message: {'role', 'content', 'tool_calls'?}."""
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        return self._post("/api/chat", payload).get("message", {})

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": self.embed_model, "input": texts})
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError("Embedding model returned an unexpected response")
        return vectors
