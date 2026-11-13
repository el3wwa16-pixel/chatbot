"""Quick check before running the app:  python check_setup.py"""
import sys

import config
from bot import db
from bot.llm import OllamaClient
from bot.rag import KnowledgeBase


def main() -> int:
    ok = True
    client = OllamaClient(config.OLLAMA_HOST, config.CHAT_MODEL, config.EMBED_MODEL)

    db.init_db(config.DB_PATH)
    print(f"[OK] Database ready: {config.DB_PATH}")

    kb = KnowledgeBase(config.KNOWLEDGE_PATH)
    print(f"[OK] Knowledge base: {len(kb.docs)} topics")

    if not client.is_up():
        print(f"[X]  Ollama is not running at {config.OLLAMA_HOST}. Open the Ollama app first.")
        return 1
    print("[OK] Ollama is running")

    if client.has_model(config.CHAT_MODEL):
        print(f"[OK] Chat model: {config.CHAT_MODEL}")
    else:
        print(f"[X]  Chat model missing. Run:  ollama pull {config.CHAT_MODEL}")
        ok = False

    if client.has_model(config.EMBED_MODEL):
        print(f"[OK] Embedding model: {config.EMBED_MODEL}")
    else:
        print(f"[!]  Embedding model missing (the bot still works with keyword search). "
              f"For better search run:  ollama pull {config.EMBED_MODEL}")

    print("\nReady. Start the app with:  streamlit run app.py" if ok
          else "\nFix the items marked [X], then run this check again.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
