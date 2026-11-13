"""Project settings. Every value can be overridden with an environment variable."""
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_PATH = DATA_DIR / "knowledge.json"
FAQ_PATH = DATA_DIR / "faq.txt"
# Files the app writes. On a host where the project folder is read-only, use the temp folder.
_WRITABLE_DIR = DATA_DIR if os.access(DATA_DIR, os.W_OK) else Path(tempfile.gettempdir())
DB_PATH = Path(os.getenv("TOURBOT_DB", str(_WRITABLE_DIR / "tourism.db")))
EMB_CACHE_PATH = _WRITABLE_DIR / "emb_cache.npz"

# Ollama runs on your own machine: no API key, no credits.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")   # weak laptop -> qwen2.5:3b
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")     # multilingual embeddings

NUM_CTX = int(os.getenv("NUM_CTX", "8192"))          # lower to 4096 if RAM is tight
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

TOP_K = 4              # knowledge chunks given to the model per question
MAX_TOOL_ROUNDS = 5    # safety limit on tool-calling loops
HISTORY_MESSAGES = 12  # how many past messages the model sees

BOT_NAME = "ديمو"  # short for Demetrius of Phalerum (ديمتريوس الفاليري)
