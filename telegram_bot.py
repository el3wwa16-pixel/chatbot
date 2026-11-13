"""Telegram channel for the same bot.  Run:  python telegram_bot.py

1. In Telegram, talk to @BotFather, send /newbot, and copy the token it gives you.
2. Set it as an environment variable named TELEGRAM_BOT_TOKEN (never put it in the code).
3. Run this file and keep it running; it answers messages while it is open.
"""
import os
import sys
import time

import requests

import config
from bot import db
from bot.agent import TourismAgent
from bot.channel import plain_answer, to_plain_text
from bot.cloud_llm import DEFAULT_BASE_URL, DEFAULT_MODEL, CloudClient
from bot.faq import FAQ
from bot.rag import KnowledgeBase
from bot.simple import SimpleAgent

WELCOME = "أهلًا بيك! أنا ديمو، رفيق رحلتك في مصر. اسألني عن أي مكان، أو الجو، أو الفنادق والجولات."


def build():
    db.init_db(config.DB_PATH)
    kb = KnowledgeBase(config.KNOWLEDGE_PATH)
    key = os.getenv("CLOUD_API_KEY") or os.getenv("GROQ_API_KEY") or ""
    cloud = CloudClient(key, model=os.getenv("CLOUD_MODEL", DEFAULT_MODEL),
                        base_url=os.getenv("CLOUD_BASE_URL", DEFAULT_BASE_URL)) if key else None
    return {"faq": FAQ(config.FAQ_PATH), "simple": SimpleAgent(kb, config.DB_PATH),
            "model": TourismAgent(cloud, kb, config.DB_PATH) if cloud else None, "history": {}}


def handle_update(update: dict, bot: dict) -> tuple[int, str] | None:
    """Returns (chat id, reply text) for a text message, or None for anything else."""
    message = update.get("message") or {}
    text, chat_id = (message.get("text") or "").strip(), (message.get("chat") or {}).get("id")
    if not text or chat_id is None:
        return None
    if text.startswith("/start"):
        return chat_id, WELCOME
    history = bot["history"].setdefault(chat_id, [])
    reply = plain_answer(text, history, bot["faq"], bot["simple"], bot["model"])
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
    del history[:-12]
    return chat_id, to_plain_text(reply)[:4000]


def main() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN first (see the top of this file).")
        return 1
    api, bot, offset = f"https://api.telegram.org/bot{token}", build(), 0
    print("Demo is listening on Telegram. Press Ctrl+C to stop.")
    while True:
        try:
            updates = requests.get(f"{api}/getUpdates", params={"offset": offset, "timeout": 30},
                                   timeout=40).json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                out = handle_update(update, bot)
                if out:
                    requests.post(f"{api}/sendMessage", json={"chat_id": out[0], "text": out[1]}, timeout=20)
        except (requests.RequestException, ValueError, KeyError):
            time.sleep(5)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
