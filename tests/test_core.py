"""Tests for everything that does not need a running model.  Run:  pytest -q"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from bot import db, tools  # noqa: E402
from bot.agent import FALLBACK_TEXT, TourismAgent  # noqa: E402
from bot.cities import canonical_city  # noqa: E402
from bot.llm import OllamaError  # noqa: E402
from bot.rag import KnowledgeBase  # noqa: E402
from bot.textutil import normalize, tokenize  # noqa: E402

FUTURE = (date.today() + timedelta(days=30)).isoformat()


@pytest.fixture
def dbp(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


class FakeClient:
    """Stands in for Ollama: replays scripted chat replies, optional fake embeddings."""
    embed_model = "fake-embed"

    def __init__(self, replies=(), embed_ok=False):
        self.replies = list(replies)
        self.embed_ok = embed_ok
        self.chat_calls = []

    def chat(self, messages, tools=None):
        self.chat_calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.replies.pop(0)

    def embed(self, texts):
        if not self.embed_ok:
            raise OllamaError("no embeddings")
        # toy embedding: one dimension per keyword
        keys = ["قايتباي", "الكرنك", "سيوة", "كشري"]
        return [[1.0 if k in t else 0.0 for k in keys] + [0.01] for t in texts]


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


# ---- text + cities ---------------------------------------------------------
def test_normalize_unifies_arabic_variants():
    assert normalize("الإسكندرية") == normalize("الاسكندريه")
    assert "قلعه" in tokenize("عايز أعرف عن القلعة")


@pytest.mark.parametrize("raw,expected", [
    ("اسكندرية", "Alexandria"), ("الإسكندرية", "Alexandria"), ("Alex", "Alexandria"),
    ("الأقصر", "Luxor"), ("شرم", "Sharm El Sheikh"), ("ASWAN", "Aswan"),
    ("مرسي علم", "Marsa Alam"), ("باريس", None), ("", None),
])
def test_canonical_city(raw, expected):
    assert canonical_city(raw) == expected


# ---- knowledge base --------------------------------------------------------
def test_knowledge_file_is_valid():
    docs = json.loads(config.KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    assert len(docs) >= 40
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids))
    for d in docs:
        assert all(d.get(k) for k in ("title_ar", "title_en", "city", "category", "text"))


@pytest.mark.parametrize("query,expected_id", [
    ("احكيلي عن قلعة قايتباي", "qaitbay"),
    ("Tell me about Karnak temple", "karnak"),
    ("ايه ارقام الطوارئ في مصر", "safety"),
    ("أروح فين في إسكندرية في يومين برنامج", "itinerary-alex-2"),
    ("abu simbel sun alignment", "abu-simbel"),
    ("واحة سيوة", "siwa"),
])
def test_keyword_search_finds_the_right_topic(query, expected_id):
    kb = KnowledgeBase(config.KNOWLEDGE_PATH)
    hits = kb.search(query, k=4)
    assert kb.mode == "keyword"
    assert expected_id in [h["doc"]["id"] for h in hits]


def test_search_falls_back_when_embeddings_fail():
    kb = KnowledgeBase(config.KNOWLEDGE_PATH, client=FakeClient(embed_ok=False))
    hits = kb.search("قلعة قايتباي")
    assert kb.mode == "keyword" and hits[0]["doc"]["id"] == "qaitbay"


def test_semantic_search_and_cache(tmp_path):
    cache = tmp_path / "emb.npz"
    kb = KnowledgeBase(config.KNOWLEDGE_PATH, client=FakeClient(embed_ok=True), cache_path=cache)
    hits = kb.search("قايتباي", k=2)
    assert kb.mode == "semantic" and cache.exists()
    assert "قايتباي" in hits[0]["doc"]["text"] + hits[0]["doc"]["title_ar"]
    # second instance loads from cache even though embedding docs would now fail
    client2 = FakeClient(embed_ok=True)
    kb2 = KnowledgeBase(config.KNOWLEDGE_PATH, client=client2, cache_path=cache)
    client2.embed_ok = False
    assert kb2.ensure_embeddings() is True


def test_empty_query_returns_nothing():
    assert KnowledgeBase(config.KNOWLEDGE_PATH).search("   ") == []


# ---- catalogue + bookings --------------------------------------------------
def test_seed_is_inserted_once(dbp):
    n = len(db.query(dbp, "SELECT * FROM hotels"))
    db.init_db(dbp)
    assert n > 10 and len(db.query(dbp, "SELECT * FROM hotels")) == n


def test_search_hotels_filters(dbp):
    out = tools.search_hotels(dbp, "الأقصر", max_price_egp=2000)
    assert out["city"] == "Luxor" and out["hotels"]
    assert all(h["price_per_night_egp"] <= 2000 for h in out["hotels"])
    assert "error" in tools.search_hotels(dbp, "باريس")


def test_search_tours(dbp):
    assert all(t["city"] == "Alexandria" for t in tools.search_tours(dbp, "Alex")["tours"])
    assert len(tools.search_tours(dbp)["tours"]) > 10


def test_hotel_booking_total_and_lifecycle(dbp):
    hotel = tools.search_hotels(dbp, "Alexandria")["hotels"][0]
    b = tools.create_booking(dbp, "hotel", hotel["id"], "Sara Ali", FUTURE, nights=3, people=3)
    assert b["status"] == "confirmed" and b["ref"].startswith("TB-")
    assert b["rooms"] == 2 and b["total_egp"] == hotel["price_per_night_egp"] * 3 * 2
    assert tools.get_booking(dbp, b["ref"].lower())["customer_name"] == "Sara Ali"
    assert tools.cancel_booking(dbp, b["ref"])["status"] == "cancelled"
    assert "error" in tools.cancel_booking(dbp, b["ref"])
    assert tools.get_booking(dbp, b["ref"])["status"] == "cancelled"


def test_tour_booking_total_accepts_string_numbers(dbp):
    tour = tools.search_tours(dbp, "Luxor")["tours"][0]
    b = tools.create_booking(dbp, "tour", str(tour["id"]), "Omar", FUTURE, people="4")
    assert b["total_egp"] == tour["price_per_person_egp"] * 4


@pytest.mark.parametrize("kwargs", [
    dict(kind="flight", item_id=1, customer_name="A", start_date=FUTURE),
    dict(kind="hotel", item_id=1, customer_name="  ", start_date=FUTURE),
    dict(kind="hotel", item_id=1, customer_name="A", start_date="next friday"),
    dict(kind="hotel", item_id=1, customer_name="A", start_date="2020-01-01"),
    dict(kind="hotel", item_id=9999, customer_name="A", start_date=FUTURE),
    dict(kind="tour", item_id=1, customer_name="A", start_date=FUTURE, people=0),
])
def test_bad_bookings_are_rejected(dbp, kwargs):
    assert "error" in tools.create_booking(dbp, **kwargs)
    assert db.list_bookings(dbp) == []


def test_unknown_booking_and_handoff(dbp):
    assert "error" in tools.get_booking(dbp, "TB-NOPE00")
    assert tools.handoff_to_human(dbp, "refund")["ticket"] == 1
    assert tools.handoff_to_human(dbp, "")["ticket"] == 2
    assert db.list_handoffs(dbp)[0]["reason"] == "unspecified"


def test_dispatch_never_raises(dbp):
    assert "error" in tools.dispatch("fly_to_moon", {}, dbp)
    assert "error" in tools.dispatch("create_booking", {"kind": "hotel"}, dbp)
    assert "error" in tools.dispatch("get_booking", {"bogus": 1}, dbp)
    assert "error" in tools.dispatch("get_booking", "TB-1", dbp)


# ---- weather (network mocked) ----------------------------------------------
class _Resp:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


def test_weather_known_city(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"], seen["params"] = url, params
        return _Resp({"current": {"temperature_2m": 31.5, "weather_code": 0, "wind_speed_10m": 9},
                      "daily": {"time": ["2026-11-13"], "temperature_2m_max": [33],
                                "temperature_2m_min": [19], "precipitation_probability_max": [0]}})
    monkeypatch.setattr(tools.requests, "get", fake_get)
    out = tools.get_weather("أسوان")
    assert out["city"] == "Aswan" and out["now"]["temperature_c"] == 31.5
    assert "forecast" in seen["url"] and round(seen["params"]["latitude"]) == 24
    assert out["forecast"][0]["max_c"] == 33


def test_weather_offline_and_unknown(monkeypatch):
    def boom(*a, **k):
        raise tools.requests.ConnectionError("offline")
    monkeypatch.setattr(tools.requests, "get", boom)
    assert "error" in tools.get_weather("Cairo")
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: _Resp({}))
    assert "error" in tools.get_weather("Atlantis")


# ---- agent loop ------------------------------------------------------------
def make_agent(dbp, replies):
    client = FakeClient(replies)
    return TourismAgent(client, KnowledgeBase(config.KNOWLEDGE_PATH), dbp), client


def test_agent_answers_from_knowledge(dbp):
    agent, client = make_agent(dbp, [{"role": "assistant", "content": "القلعة بناها قايتباي."}])
    result = agent.reply([], "احكيلي عن قلعة قايتباي")
    assert result.text == "القلعة بناها قايتباي." and "قلعة قايتباي" in result.sources
    system = client.chat_calls[0]["messages"][0]
    assert system["role"] == "system" and "فاروس" in system["content"]
    assert date.today().isoformat() in system["content"]
    assert client.chat_calls[0]["tools"]


def test_agent_runs_tools_then_answers(dbp):
    hotel_id = tools.search_hotels(dbp, "Luxor")["hotels"][0]["id"]
    agent, client = make_agent(dbp, [
        {"role": "assistant", "content": "", "tool_calls": [call("search_hotels", city="الأقصر")]},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {
            "name": "create_booking", "arguments": json.dumps({
                "kind": "hotel", "item_id": hotel_id, "customer_name": "Mona",
                "start_date": FUTURE, "nights": 2, "people": 2})}}]},
        {"role": "assistant", "content": "تم الحجز."},
    ])
    history = [{"role": "user", "content": "عايز فندق في الأقصر"},
               {"role": "assistant", "content": "تمام، باسم مين؟"}]
    result = agent.reply(history, "باسم Mona، أكد الحجز")
    assert result.text == "تم الحجز."
    assert [e["tool"] for e in result.tool_events] == ["search_hotels", "create_booking"]
    booking = db.list_bookings(dbp)[0]
    assert booking["customer_name"] == "Mona" and booking["nights"] == 2
    last = client.chat_calls[-1]["messages"]
    assert last[-1]["role"] == "tool" and booking["ref"] in last[-1]["content"]
    assert [m["role"] for m in last[:4]] == ["system", "user", "assistant", "user"]


def test_agent_survives_bad_tool_call_and_empty_reply(dbp):
    agent, _ = make_agent(dbp, [
        {"role": "assistant", "content": "", "tool_calls": [call("nonexistent_tool", x=1)]},
        {"role": "assistant", "content": ""},
    ])
    result = agent.reply([], "hello")
    assert result.text == FALLBACK_TEXT and "error" in result.tool_events[0]["result"]


def test_agent_stops_endless_tool_loops(dbp):
    loop = {"role": "assistant", "content": "", "tool_calls": [call("search_tours", city="Cairo")]}
    agent, client = make_agent(dbp, [loop] * config.MAX_TOOL_ROUNDS
                               + [{"role": "assistant", "content": "done"}])
    assert agent.reply([], "tours?").text == "done"
    assert client.chat_calls[-1]["tools"] is None
    assert len(client.chat_calls) == config.MAX_TOOL_ROUNDS + 1


# ---- simple mode (no model) --------------------------------------------------
from bot.simple import SimpleAgent, format_booking, is_english  # noqa: E402


@pytest.fixture
def simple(dbp):
    return SimpleAgent(KnowledgeBase(config.KNOWLEDGE_PATH), dbp)


def test_simple_knowledge_answer(simple):
    r = simple.reply([], "احكيلي عن قلعة قايتباي")
    assert "فاروس" in r.text and r.sources[0] == "قلعة قايتباي" and not r.tool_events


def test_simple_itinerary_goes_to_knowledge_not_tours(simple):
    r = simple.reply([], "اعملي برنامج رحلة يومين في إسكندرية")
    assert "اليوم الأول" in r.text and not r.tool_events


def test_simple_english(simple):
    assert is_english("What tours can I book in Luxor?")
    r = simple.reply([], "What tours can I book in Luxor?")
    assert r.tool_events[0]["tool"] == "search_tours" and "Tours in Luxor" in r.text
    assert "Karnak" in simple.reply([], "Tell me about Karnak temple").text


def test_simple_hotels_with_budget_and_arabic_digits(simple):
    r = simple.reply([], "عايز فندق في الأقصر بأقل من ٢٠٠٠ جنيه")
    assert r.tool_events[0]["args"] == {"city": "Luxor", "max_price_egp": 2000}
    assert "1,300" in r.text and "2,600" not in r.text
    assert "في أي مدينة" in simple.reply([], "عايز فندق").text


def test_simple_remembers_city_from_history(simple, monkeypatch):
    monkeypatch.setattr(tools, "get_weather", lambda city: {
        "city": city, "now": {"temperature_c": 30, "condition": "صافٍ", "wind_kmh": 5},
        "forecast": [{"date": "2026-11-14", "max_c": 31, "min_c": 18, "rain_chance_percent": 0}]})
    history = [{"role": "user", "content": "احكيلي عن أسوان"},
               {"role": "assistant", "content": "..."}]
    r = simple.reply(history, "الجو عامل ايه هناك؟")
    assert r.tool_events[0]["args"] == {"city": "Aswan"} and "أسوان" in r.text and "30" in r.text
    assert "في أي مدينة" in simple.reply([], "الجو عامل ايه؟").text


def test_simple_tour_word_is_not_weather(simple):
    r = simple.reply([], "عايز اعرف الجولات في أسوان")
    assert r.tool_events[0]["tool"] == "search_tours"


def test_simple_booking_lookup_cancel_and_handoff(simple, dbp):
    b = tools.create_booking(dbp, "tour", 1, "Ali", FUTURE, people=2)
    assert b["ref"] in simple.reply([], f"عايز اعرف حجزي {b['ref'].lower()}").text
    assert "تم إلغاء" in simple.reply([], f"الغي الحجز {b['ref']}").text
    assert tools.get_booking(dbp, b["ref"])["status"] == "cancelled"
    assert "مفيش حجز" in simple.reply([], "حجزي TB-ZZZZZZ").text
    assert "#1" in simple.reply([], "عندي شكوى وعايز أكلم موظف").text
    assert "حجز سريع" in simple.reply([], "عايز احجز").text


def test_simple_greeting_and_unknown(simple):
    assert "أهلًا" in simple.reply([], "اهلا").text
    assert "ملقتش" in simple.reply([], "xyzzy qwerty").text or simple.reply([], "xyzzy qwerty").text


def test_format_booking(dbp):
    b = tools.create_booking(dbp, "hotel", 1, "Sara", FUTURE, nights=2, people=2)
    assert b["ref"] in format_booking(b) and "الليالي: 2" in format_booking(b)
    assert "nights: 2" in format_booking(tools.get_booking(dbp, b["ref"]), en=True)


@pytest.mark.parametrize("query,expected_id", [
    ("أروح فين في إسكندرية في يومين؟", "itinerary-alex-2"),
    ("اعمل ايه في القاهرة 3 ايام", "itinerary-cairo-3"),
    ("ايه احسن وقت ازور مصر", "best-time"),
    ("محتاج فيزا؟", "visa"),
    ("اكل ايه في اسكندرية", "alex-food"),
    ("Is Egypt safe? emergency numbers", "safety"),
    ("how do I get around, is there a metro", "transport"),
    ("تذاكر الاهرامات بكام", "tickets"),
    ("what is the best time to visit", "best-time"),
])
def test_simple_mode_top_answer(simple, query, expected_id):
    kb = simple.kb
    top = kb.search(query, k=3, prefer_city=canonical_city(query))[0]["doc"]["id"]
    assert top == expected_id
