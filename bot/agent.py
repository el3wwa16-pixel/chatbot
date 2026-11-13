"""The agent: retrieve knowledge, let the model answer or call tools, loop until done."""
import json
from dataclasses import dataclass, field
from datetime import date

import config
from .cities import CITIES, canonical_city
from .tools import TOOL_SPECS, dispatch

SYSTEM_PROMPT = """You are {bot_name} (Rafiq), a friendly AI travel assistant for tourism in Egypt.
Today's date is {today}.

LANGUAGE
- Always reply in the language of the customer's last message. If they write
  Egyptian Arabic, reply in natural Egyptian Arabic; English gets English.

WHAT YOU DO
1. Tour guide: answer questions about places, history, food and practical tips.
2. Trip planner: build day-by-day itineraries that fit the customer's days, interests and budget.
3. Booking assistant: find and book hotels and tours with the tools.
4. Customer service: look up or cancel bookings, and hand off to a human when needed.

RULES
- For facts about places, rely on the KNOWLEDGE section below. If the answer is not
  there and you are not sure, say you are not sure. Never invent ticket prices or
  opening hours; tell the customer to check the official source because these change.
- Use get_weather for any weather question. Never guess the weather.
- Hotels, tours and prices come ONLY from search_hotels / search_tours. Never invent them.
- Before create_booking you need: the item, the customer's name, the start date,
  number of people (and nights for hotels). Ask for whatever is missing, summarise
  the booking with its total, and call create_booking only after the customer says yes.
- After a booking, give the reference code and say it is a demo booking with no real payment.
- Never ask for card numbers or passwords.
- For complaints, refund requests, emergencies or anything you cannot solve, call
  handoff_to_human and tell the customer a human agent will follow up.
  Emergency numbers in Egypt: tourist police 126, ambulance 123, police 122.
- Bookable cities: {cities}.
- Keep answers clear and reasonably short; use short lists for itineraries.

KNOWLEDGE
{knowledge}
"""

FALLBACK_TEXT = "معلش، مقدرتش أجهّز رد دلوقتي. ممكن تعيد صياغة سؤالك؟"


@dataclass
class AgentResult:
    text: str
    sources: list[str] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    search_mode: str = "keyword"


class TourismAgent:
    def __init__(self, client, kb, db_path):
        self.client = client
        self.kb = kb
        self.db_path = db_path

    def _system(self, hits: list[dict]) -> str:
        if hits:
            knowledge = "\n\n".join(
                f"[{i}] {h['doc'].get('title_ar', '')} ({h['doc'].get('title_en', '')}) "
                f"- {h['doc'].get('city', '')}\n{h['doc'].get('text', '')}"
                for i, h in enumerate(hits, 1))
        else:
            knowledge = "(nothing relevant found in the knowledge base)"
        return SYSTEM_PROMPT.format(bot_name=config.BOT_NAME, today=date.today().isoformat(),
                                    cities=", ".join(CITIES), knowledge=knowledge)

    def reply(self, history: list[dict], user_text: str) -> AgentResult:
        """history: earlier [{'role': 'user'|'assistant', 'content': str}] messages."""
        hits = self.kb.search(user_text, k=config.TOP_K,
                              prefer_city=canonical_city(user_text))
        result = AgentResult(text="", search_mode=self.kb.mode,
                             sources=[h["doc"].get("title_ar", "") for h in hits])

        messages = [{"role": "system", "content": self._system(hits)}]
        messages += [{"role": m["role"], "content": m["content"]}
                     for m in history[-config.HISTORY_MESSAGES:]
                     if m.get("role") in ("user", "assistant") and m.get("content")]
        messages.append({"role": "user", "content": user_text})

        for _ in range(config.MAX_TOOL_ROUNDS):
            msg = self.client.chat(messages, tools=TOOL_SPECS)
            calls = msg.get("tool_calls") or []
            if not calls:
                result.text = (msg.get("content") or "").strip() or FALLBACK_TEXT
                return result
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                name, args = fn.get("name", ""), fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                output = dispatch(name, args, self.db_path)
                result.tool_events.append({"tool": name, "args": args, "result": output})
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(output, ensure_ascii=False)})

        # Too many tool rounds: ask for a final answer without tools.
        msg = self.client.chat(messages)
        result.text = (msg.get("content") or "").strip() or FALLBACK_TEXT
        return result
