"""One question in, one text answer out: the same brain for any channel (Telegram, tests)."""
import re

from .llm import OllamaError


def plain_answer(question: str, history: list[dict], faq, simple_agent, model_agent=None) -> str:
    stored = faq.match(question)
    if stored:
        return stored
    if model_agent is not None:
        try:
            return model_agent.reply(history, question).text
        except OllamaError:
            pass  # model unavailable: fall back to the simple mode
    return simple_agent.reply(history, question).text


def to_plain_text(markdown: str) -> str:
    """Strip the little Markdown the bot uses, for channels that show raw text."""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1: \2", markdown)
    return re.sub(r"\*\*|__|`", "", text)
