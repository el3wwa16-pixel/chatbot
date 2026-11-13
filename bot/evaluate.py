"""Automatic accuracy check: runs a fixed set of questions through the bot's
no-model pipeline and reports how many were routed and answered correctly.

It measures retrieval and routing (right topic, right stored answer, right tool),
not the wording of a model's reply, and it never touches the network: weather and
Wikipedia calls are recorded and skipped.
"""
import json
import tempfile
from pathlib import Path

from . import db
from .cities import canonical_city
from .simple import LOCAL_MATCH, SimpleAgent
from .tools import dispatch

_NETWORK_TOOLS = {"get_weather", "search_web_knowledge"}


def _offline_dispatch(name, args, db_path):
    if name in _NETWORK_TOOLS:
        return {"error": "skipped in the accuracy check"}
    return dispatch(name, args, db_path)


def run(kb, faq, cases_path) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    scratch = Path(tempfile.mkdtemp()) / "eval.db"  # bookings made here never reach the real data
    db.init_db(scratch)
    agent = SimpleAgent(kb, scratch, dispatcher=_offline_dispatch)
    rows = []
    for case in cases:
        question, kind, expect = case["q"], case["type"], case["expect"]
        stored = faq.match(question)
        if kind == "faq":
            got = stored or ""
            ok = expect in got
            shown = got[:60] or "no stored answer matched"
        elif kind == "topic":
            hits = kb.search(question, k=1, prefer_city=canonical_city(question))
            top = hits[0]["doc"] if hits else None
            strong = bool(top) and kb.match_strength(question, top) >= LOCAL_MATCH
            ok = not stored and strong and top["id"] == expect
            shown = "answered from the FAQ instead" if stored else (top["id"] if top else "nothing found") \
                + ("" if strong else " (weak match)")
        else:
            called = [] if stored else [e["tool"] for e in agent.reply([], question).tool_events]
            ok = expect in called
            shown = "answered from the FAQ instead" if stored else (", ".join(called) or "no tool called")
        rows.append({"question": question, "type": kind, "expected": expect, "got": shown, "ok": ok})

    by_type = {}
    for row in rows:
        entry = by_type.setdefault(row["type"], {"total": 0, "passed": 0})
        entry["total"] += 1
        entry["passed"] += row["ok"]
    passed = sum(r["ok"] for r in rows)
    return {"total": len(rows), "passed": passed,
            "accuracy": round(100 * passed / len(rows), 1) if rows else 0.0,
            "by_type": by_type, "failures": [r for r in rows if not r["ok"]], "rows": rows}
