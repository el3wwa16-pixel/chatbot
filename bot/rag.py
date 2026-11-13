"""Knowledge base search (the R in RAG).

Semantic search with local multilingual embeddings when Ollama's embedding
model is available; automatic fallback to BM25 keyword search when it is not,
so the bot never stops working.
"""
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .llm import OllamaError
from .textutil import tokenize


class _BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(t) for t in docs_tokens]
        self.lengths = [len(t) for t in docs_tokens]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if docs_tokens else 0.0
        df = Counter()
        for t in docs_tokens:
            df.update(set(t))
        n = len(docs_tokens)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            for w in query_tokens:
                f = tf.get(w, 0)
                if not f:
                    continue
                norm = f + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / norm
            out.append(s)
        return out


class KnowledgeBase:
    RETRY_SECONDS = 60

    def __init__(self, path, client=None, cache_path=None):
        self.path = Path(path)
        self.client = client
        self.cache_path = Path(cache_path) if cache_path else None
        raw = self.path.read_bytes()
        self.docs: list[dict] = json.loads(raw.decode("utf-8"))
        self._hash = hashlib.sha1(raw).hexdigest()
        self._bm25 = _BM25([tokenize(self._doc_text(d)) for d in self.docs])
        self._emb: np.ndarray | None = None
        self._last_fail = 0.0

    @staticmethod
    def _doc_text(d: dict) -> str:
        return " ".join(str(d.get(k, "")) for k in
                        ("title_ar", "title_en", "city", "category", "keywords", "text"))

    @property
    def mode(self) -> str:
        return "semantic" if self._emb is not None else "keyword"

    # ---- embeddings ---------------------------------------------------
    def _cache_key(self) -> str:
        model = getattr(self.client, "embed_model", "")
        return f"{self._hash}:{model}"

    def _load_cache(self) -> bool:
        if not self.cache_path or not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            if str(data["key"]) == self._cache_key() and len(data["emb"]) == len(self.docs):
                self._emb = data["emb"]
                return True
        except (OSError, KeyError, ValueError):
            pass
        return False

    def ensure_embeddings(self) -> bool:
        """Build (or load) document embeddings. Returns True when semantic search is on."""
        if self._emb is not None:
            return True
        if self.client is None or time.time() - self._last_fail < self.RETRY_SECONDS:
            return False
        if self._load_cache():
            return True
        try:
            vectors = []
            texts = [self._doc_text(d) for d in self.docs]
            for i in range(0, len(texts), 16):
                vectors.extend(self.client.embed(texts[i:i + 16]))
            emb = self._normalise(np.asarray(vectors, dtype=np.float32))
        except (OllamaError, ValueError):
            self._last_fail = time.time()
            return False
        self._emb = emb
        if self.cache_path:
            try:
                np.savez(self.cache_path, emb=emb, key=np.array(self._cache_key()))
            except OSError:
                pass
        return True

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    # ---- search -------------------------------------------------------
    def search(self, query: str, k: int = 4, prefer_city: str | None = None) -> list[dict]:
        """Returns up to k results: {'doc': {...}, 'score': float}.

        prefer_city: when the question names a city, topics about that city get
        a 25% score boost (so "a plan for Alexandria" does not return the Cairo plan).
        """
        if not query or not query.strip():
            return []
        hits = self._ranked(query, max(2 * k, 8) if prefer_city else k)
        if prefer_city:
            for h in hits:
                if h["doc"].get("city") == prefer_city:
                    h["score"] *= 1.25
            hits.sort(key=lambda h: -h["score"])
        return hits[:k]

    def _ranked(self, query: str, k: int) -> list[dict]:
        if self.ensure_embeddings():
            try:
                q = self._normalise(np.asarray(self.client.embed([query])[0], dtype=np.float32))
                sims = self._emb @ q
                order = np.argsort(-sims)[:k]
                return [{"doc": self.docs[i], "score": float(sims[i])} for i in order]
            except (OllamaError, ValueError):
                pass  # fall through to keyword search
        scores = self._bm25.scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"doc": self.docs[i], "score": float(scores[i])} for i in order if scores[i] > 0]
