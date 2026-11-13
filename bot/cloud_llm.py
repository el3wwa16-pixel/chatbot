"""Optional cloud model through any OpenAI-compatible API (Groq by default).

Used only when an API key is configured; the hosted site cannot run a local
model, so this is what gives it free conversation there. The client exposes the
same chat() interface as OllamaClient, so the agent does not care which one it has.
"""
import json

import requests

from .llm import OllamaError

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_STT_MODEL = "whisper-large-v3"


class CloudError(OllamaError):
    """Raised when the cloud model is unreachable or refuses the request."""


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert the agent's (Ollama-style) messages to the OpenAI wire format."""
    out, pending_ids, counter = [], [], 0
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls, pending_ids = [], []
            for call in m["tool_calls"]:
                fn = call.get("function") or {}
                counter += 1
                call_id = call.get("id") or f"call_{counter}"
                pending_ids.append(call_id)
                args = fn.get("arguments") or {}
                calls.append({"id": call_id, "type": "function", "function": {
                    "name": fn.get("name", ""),
                    "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            counter += 1
            call_id = pending_ids.pop(0) if pending_ids else f"call_{counter}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class CloudClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 stt_model: str = DEFAULT_STT_MODEL, temperature: float = 0.3, timeout: int = 60):
        self.api_key = api_key
        self.chat_model = model
        self.base_url = base_url.rstrip("/")
        self.stt_model = stt_model
        self.temperature = temperature
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _check(self, response) -> dict:
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {})
                detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            except ValueError:
                detail = response.text[:200]
            raise CloudError(f"Cloud model error ({response.status_code}): {detail}")
        try:
            return response.json()
        except ValueError as e:
            raise CloudError("Cloud model returned an unreadable response") from e

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        payload = {"model": self.chat_model, "messages": to_openai_messages(messages),
                   "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              headers=self._headers(), timeout=self.timeout)
        except requests.RequestException as e:
            raise CloudError("Cloud model is not reachable") from e
        try:
            msg = self._check(r)["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise CloudError("Cloud model returned an unexpected response") from e
        calls = []
        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"id": call.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            out["tool_calls"] = calls
        return out

    def transcribe(self, audio: bytes, filename: str = "voice.wav") -> str:
        """Speech to text for voice messages."""
        try:
            r = requests.post(f"{self.base_url}/audio/transcriptions", headers=self._headers(),
                              files={"file": (filename, audio)}, data={"model": self.stt_model},
                              timeout=self.timeout)
        except requests.RequestException as e:
            raise CloudError("Speech service is not reachable") from e
        return str(self._check(r).get("text", "")).strip()
