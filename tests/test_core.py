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

    def fake_get(url, params=None, timeout=None, headers=None):
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


def test_weather_falls_back_to_second_provider(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        if "open-meteo" in url:
            return _Resp({"error": True, "reason": "limit exceeded"})
        return _Resp({"current_condition": [{"temp_C": "29", "windspeedKmph": "11",
                                             "weatherDesc": [{"value": "Sunny"}]}],
                      "weather": [{"date": "2026-11-13", "maxtempC": "31", "mintempC": "17",
                                   "hourly": [{"chanceofrain": "0"}, {"chanceofrain": "5"}]}]})
    monkeypatch.setattr(tools.requests, "get", fake_get)
    out = tools.get_weather("Aswan")
    assert out["source"] == "wttr.in" and out["now"]["temperature_c"] == 29.0
    assert out["forecast"][0] == {"date": "2026-11-13", "max_c": 31.0, "min_c": 17.0,
                                  "rain_chance_percent": 5}


# ---- external sources (network mocked) ---------------------------------------
from bot import web_sources  # noqa: E402


def _wiki(pages):
    return _Resp({"query": {"pages": {str(i): p for i, p in enumerate(pages)}}})


def test_clean_query():
    assert web_sources.clean_query("احكيلي عن معبد دندرة") == "معبد دندرة"
    assert web_sources.clean_query("مين هو نجيب محفوظ؟") == "نجيب محفوظ"
    assert web_sources.clean_query("Tell me about the Eiffel Tower") == "Eiffel Tower"
    assert web_sources.clean_query("ايه ده") == "ايه ده"


def test_web_search_picks_first_real_page_and_falls_back(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        if "ar.wikipedia" in url:
            return _wiki([{"index": 1, "title": "توضيح", "extract": "x" * 80,
                           "pageprops": {"disambiguation": ""}},
                          {"index": 2, "title": "قصير", "extract": "قصير"}])
        return _wiki([{"index": 2, "title": "Other", "extract": "o" * 80, "fullurl": "u2"},
                      {"index": 1, "title": "Dendera", "extract": "d" * 80, "fullurl": "u1"}])
    monkeypatch.setattr(web_sources.requests, "get", fake_get)
    out = web_sources.search_web_knowledge("احكيلي عن معبد دندرة", lang="ar")
    assert out["title"] == "Dendera" and out["source"] == "Wikipedia" and out["url"] == "u1"
    assert out["query"] == "معبد دندرة" and len(calls) == 2


def test_web_search_errors(monkeypatch):
    def boom(*a, **k):
        raise web_sources.requests.ConnectionError("offline")
    monkeypatch.setattr(web_sources.requests, "get", boom)
    assert "unavailable" in web_sources.search_web_knowledge("Luxor", lang="en")["error"]
    monkeypatch.setattr(web_sources.requests, "get", lambda *a, **k: _Resp({"batchcomplete": ""}))
    assert "Nothing found" in web_sources.search_web_knowledge("zzzz", lang="en")["error"]
    assert "error" in web_sources.search_web_knowledge("  ")


def test_match_strength_separates_known_from_unknown():
    kb = KnowledgeBase(config.KNOWLEDGE_PATH)
    def strength(q):
        hits = kb.search(q, k=1, prefer_city=canonical_city(q))
        return kb.match_strength(q, hits[0]["doc"]) if hits else 0.0
    for q in ["احكيلي عن قلعة قايتباي", "أروح فين في إسكندرية في يومين؟", "محتاج فيزا؟",
              "ايه احسن وقت ازور مصر", "Tell me about Karnak temple", "اكل ايه في اسكندرية",
              "Is Egypt safe? emergency numbers", "واحة سيوة"]:
        assert strength(q) >= 0.45, q
    for q in ["مين هو نجيب محفوظ", "احكيلي عن برج ايفل", "ما هي عاصمة فرنسا",
              "Who built the Eiffel Tower?", "تاريخ الدولة الفاطمية"]:
        assert strength(q) < 0.45, q


def test_simple_uses_web_for_unknown_topics_only(simple, monkeypatch):
    asked = []

    def fake_web(query, lang="ar"):
        asked.append((query, lang))
        return {"source": "ويكيبيديا العربية", "query": query, "title": "نجيب محفوظ",
                "extract": "روائي مصري حائز على جائزة نوبل.", "url": "https://ar.wikipedia.org/x"}
    monkeypatch.setattr(tools, "search_web_knowledge", fake_web)
    r = simple.reply([], "مين هو نجيب محفوظ؟")
    assert "نوبل" in r.text and "المصدر" in r.text and r.sources == ["ويكيبيديا العربية: نجيب محفوظ"]
    assert asked == [("مين هو نجيب محفوظ؟", "ar")]
    simple.reply([], "احكيلي عن قلعة قايتباي")
    assert len(asked) == 1  # known topic: answered locally, no web call


def test_simple_falls_back_to_local_when_web_fails(simple, monkeypatch):
    monkeypatch.setattr(tools, "search_web_knowledge", lambda query, lang="ar": {"error": "x"})
    assert "could not find" in simple.reply([], "xyzzy qwerty").text
    assert "ملقتش" in simple.reply([], "زززز ششش").text
    assert "معبد" in simple.reply([], "احكيلي عن معبد دندرة").text  # weak local topic still shown


def test_dispatch_routes_web_tool(dbp, monkeypatch):
    monkeypatch.setattr(tools, "search_web_knowledge", lambda query, lang="ar": {"title": query})
    assert tools.dispatch("search_web_knowledge", {"query": "Luxor"}, dbp) == {"title": "Luxor"}
    assert any(s["function"]["name"] == "search_web_knowledge" for s in tools.TOOL_SPECS)


# ---- frequently asked questions ------------------------------------------------
from bot import faq as faq_module  # noqa: E402


def test_faq_file_parses_and_answers():
    f = faq_module.FAQ(config.FAQ_PATH)
    assert len(f.entries) >= 8 and all(e["questions"] and e["answer"] for e in f.entries)
    assert "126" in f.match("رقم شرطة السياحة كام؟")
    assert "126" in f.match("لو سمحت ايه ارقام الطوارئ في مصر")
    assert "تجريبي" in f.match("هل الحجز حقيقي")
    assert "TB" in f.match("ازاي الغي الحجز")
    assert "ديمتريوس" in f.match("ليه اسمك ديمو")
    assert "التأشيرة" in f.match("محتاج فيزا؟") and "رفيق" in f.match("بتعمل ايه")
    assert "126" in f.match("Emergency numbers in Egypt?")
    assert len(f.questions(3)) == 3 and f.questions()[0] == f.entries[0]["questions"][0]


@pytest.mark.parametrize("text", [
    "احكيلي عن قلعة قايتباي", "الجو عامل إيه في أسوان؟", "عايز فندق في الأقصر بأقل من 2000 جنيه",
    "أروح فين في إسكندرية في يومين؟", "What tours can I book in Luxor?", "الغي الحجز TB-A1B2C3",
    "مين هو نجيب محفوظ", "", "؟؟",
])
def test_faq_does_not_hijack_other_questions(text):
    assert faq_module.FAQ(config.FAQ_PATH).match(text) is None


def test_faq_parser_is_forgiving(tmp_path):
    path = tmp_path / "faq.txt"
    path.write_text("# note\n\nس: سؤال اول\nس :صيغة تانية\nج: سطر واحد\nسطر اتنين\n\nفقرة جديدة\n"
                    "س: سؤال بدون اجابة\n\nQ: english one\nA: yes\n\nج: اجابة بدون سؤال\n", encoding="utf-8")
    entries = faq_module.parse(path.read_text(encoding="utf-8"))
    assert [e["questions"] for e in entries] == [["سؤال اول", "صيغة تانية"], ["english one"]]
    assert entries[0]["answer"] == "سطر واحد\nسطر اتنين\n\nفقرة جديدة" and entries[1]["answer"] == "yes"
    assert faq_module.FAQ(tmp_path / "missing.txt").match("anything") is None


# ---- extras: planner, budget, currency, photos, ticket, statistics, cloud model --------
from bot import extras, planner  # noqa: E402
from bot.cloud_llm import CloudClient, CloudError, to_openai_messages  # noqa: E402
from bot.i18n import TEXT, t  # noqa: E402


def _docs():
    return json.loads(config.KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def test_knowledge_geo_and_stops_are_consistent():
    docs = _docs()
    ids = {d["id"] for d in docs}
    assert len(docs) >= 60
    for d in docs:
        if "lat" in d:
            assert 21 < d["lat"] < 32 and 24 < d["lon"] < 37 and d.get("wiki"), d["id"]
        assert set(d.get("stops", [])) <= ids
    assert KnowledgeBase(config.KNOWLEDGE_PATH).search("احكيلي عن معبد دندرة")[0]["doc"]["id"] == "dendera"


def test_planner_builds_days_from_interests(dbp):
    plan = planner.build_plan(_docs(), dbp, "Alexandria", 2, ["history"])
    assert len(plan["days"]) == 2 and all(len(day) == 3 for day in plan["days"]) and plan["enough"]
    assert [i["slot"] for i in plan["days"][0]] == [0, 1, 2]
    titles = [i["title"] for day in plan["days"] for i in day]
    assert "قلعة قايتباي" in titles and len(set(titles)) == len(titles)
    assert plan["points"] and all(p["lat"] for p in plan["points"]) and plan["city_ar"] == "الإسكندرية"
    food_first = planner.build_plan(_docs(), dbp, "Alexandria", 1, ["food"])["days"][0][0]["title"]
    assert food_first == "أكل الإسكندرية"


def test_planner_handles_small_cities_and_limits(dbp):
    plan = planner.build_plan(_docs(), dbp, "Hurghada", 7, [])
    assert len(plan["days"]) == 7 and plan["days"][0] and not plan["enough"] and plan["days"][-1] == []
    assert any(i["kind"] == "tour" for day in plan["days"] for i in day)
    assert len(planner.build_plan(_docs(), dbp, "Cairo", 99, ["history"])["days"]) == 7
    cairo = [i["title"] for d in planner.build_plan(_docs(), dbp, "Cairo", 7, ["history"])["days"] for i in d]
    assert "أهرامات الجيزة وأبو الهول" in cairo  # Giza counts as part of a Cairo trip


def test_trip_budget(dbp):
    hotel = tools.search_hotels(dbp, "Luxor")["hotels"][0]
    tour = tools.search_tours(dbp, "Luxor")["tours"][0]
    b = extras.trip_budget(dbp, hotel["id"], 3, 3, [tour["id"]])
    assert b["total_egp"] == hotel["price_per_night_egp"] * 3 * 2 + tour["price_per_person_egp"] * 3
    assert b["per_person_egp"] == round(b["total_egp"] / 3) and len(b["lines"]) == 2
    assert extras.trip_budget(dbp, None, 3, 2, [])["total_egp"] == 0
    assert extras.trip_budget(dbp, 9999, 2, 2, [9999]) == {"lines": [], "total_egp": 0, "per_person_egp": 0}


def test_currency(monkeypatch):
    monkeypatch.setattr(extras.requests, "get", lambda *a, **k: _Resp(
        {"result": "success", "time_last_update_utc": "Fri, 13 Nov 2026 00:02:31 +0000",
         "rates": {"USD": 1, "EUR": 0.9, "EGP": 50, "GBP": 0.8, "XXX": 3}}))
    out = extras.fetch_rates()
    assert out["rates"]["EGP"] == 50 and "XXX" not in out["rates"] and out["updated"].startswith("Fri")
    assert extras.convert(100, "USD", "EGP", out["rates"]) == 5000
    assert extras.convert(90, "EUR", "EGP", out["rates"]) == 5000
    assert extras.convert(1, "USD", "JPY", out["rates"]) is None
    monkeypatch.setattr(extras.requests, "get", lambda *a, **k: _Resp({"result": "error"}))
    assert "error" in extras.fetch_rates()


def test_place_image(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None, headers=None):
        seen.update(params)
        return _Resp({"query": {"pages": {"1": {"thumbnail": {"source": "https://img/x.jpg"},
                                                "fullurl": "https://en.wikipedia.org/wiki/Karnak"}}}})
    monkeypatch.setattr(extras.requests, "get", fake_get)
    assert extras.place_image("Karnak") == {"url": "https://img/x.jpg",
                                            "page": "https://en.wikipedia.org/wiki/Karnak"}
    assert seen["pilicense"] == "free"  # only freely licensed photos
    monkeypatch.setattr(extras.requests, "get", lambda *a, **k: _Resp({"query": {"pages": {"-1": {}}}}))
    assert extras.place_image("Nothing") is None and extras.place_image("") is None


def test_ticket_html_escapes_and_lists_details(dbp):
    b = tools.create_booking(dbp, "hotel", 1, "<b>Sara</b>", FUTURE, nights=2, people=2)
    page = extras.ticket_html(b)
    assert b["ref"] in page and "&lt;b&gt;Sara&lt;/b&gt;" in page and "<b>Sara</b>" not in page
    assert "عدد الليالي" in page and "تجريبي" in page
    tour = tools.create_booking(dbp, "tour", 1, "Ali", FUTURE, people=2)
    assert "عدد الليالي" not in extras.ticket_html(tools.get_booking(dbp, tour["ref"]))


def test_phrasebook_and_interface_text():
    assert len(extras.PHRASES) >= 25 and all(len(p) == 3 and all(p) for p in extras.PHRASES)
    assert all(v.get("ar") and v.get("en") for v in TEXT.values())
    assert t("book", "en") == "Book" and t("book") == "احجز" and t("missing_key") == "missing_key"
    assert "ديمو" in t("welcome", "ar", name="ديمو")
    for page in ["chat", "plan", "budget", "money", "phrases", "stats"]:
        assert f"nav_{page}" in TEXT
    for source in db.SOURCES:
        assert f"src_{source}" in TEXT


def test_statistics(dbp):
    for text, source in [("الجو", "tool"), ("الجو", "tool"), ("قايتباي", "knowledge"),
                         ("xyz", "none"), ("xyz", "bogus")]:
        db.log_question(dbp, text, source)
    db.log_feedback(dbp, "قايتباي", 1)
    db.log_feedback(dbp, "xyz", 0)
    tools.create_booking(dbp, "tour", 1, "Ali", FUTURE)
    s = db.stats(dbp)
    assert (s["questions"], s["unanswered"], s["bookings"], s["likes"], s["dislikes"]) == (5, 2, 1, 1, 1)
    assert s["by_source"] == {"tool": 2, "knowledge": 1, "none": 2}
    assert s["top_questions"][0] == {"text": "xyz", "n": 2} or s["top_questions"][0]["n"] == 2
    assert s["unanswered_list"] == [{"text": "xyz", "n": 2}] and s["disliked_list"][0]["text"] == "xyz"


def test_simple_agent_marks_unanswered(simple, monkeypatch):
    monkeypatch.setattr(tools, "search_web_knowledge", lambda query, lang="ar": {"error": "x"})
    assert simple.reply([], "زززز ششش").answered is False
    assert simple.reply([], "احكيلي عن قلعة قايتباي").answered is True


def test_faq_buttons_follow_language():
    f = faq_module.FAQ(config.FAQ_PATH)
    assert f.questions(1, "en") == ["What can you do?"] and f.questions(1, "ar") == ["ديمو بيعمل إيه؟"]


def test_cloud_message_conversion():
    out = to_openai_messages([
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "abc", "function": {"name": "get_weather", "arguments": {"city": "Aswan"}}},
            {"function": {"name": "search_tours", "arguments": {}}}]},
        {"role": "tool", "tool_name": "get_weather", "content": "{}"},
        {"role": "tool", "tool_name": "search_tours", "content": "{}"}])
    calls = out[2]["tool_calls"]
    assert out[2]["content"] is None and calls[0]["id"] == "abc" and calls[0]["type"] == "function"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Aswan"}
    assert [m["tool_call_id"] for m in out[3:]] == ["abc", calls[1]["id"]] and "tool_name" not in out[3]


class _Post:
    def __init__(self, status, data):
        self.status_code, self._data, self.text = status, data, str(data)

    def json(self):
        return self._data


def test_cloud_client_chat_and_agent(dbp, monkeypatch):
    from bot import cloud_llm
    sent = []
    replies = [
        _Post(200, {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_hotels", "arguments": '{"city": "Luxor"}'}}]}}]}),
        _Post(200, {"choices": [{"message": {"content": "دي فنادق الأقصر."}}]}),
    ]

    def fake_post(url, json=None, headers=None, timeout=None, files=None, data=None):
        sent.append({"url": url, "json": json, "headers": headers})
        return replies.pop(0)
    monkeypatch.setattr(cloud_llm.requests, "post", fake_post)
    client = CloudClient("key123", model="m")
    agent = TourismAgent(client, KnowledgeBase(config.KNOWLEDGE_PATH), dbp)
    result = agent.reply([], "فنادق في الأقصر")
    assert result.text == "دي فنادق الأقصر." and result.tool_events[0]["tool"] == "search_hotels"
    assert sent[0]["url"].endswith("/chat/completions") and sent[0]["headers"]["Authorization"] == "Bearer key123"
    assert sent[0]["json"]["tools"] and sent[0]["json"]["model"] == "m"
    second = sent[1]["json"]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert second[-2]["tool_calls"][0]["id"] == "c1"


def test_cloud_client_errors_and_transcribe(monkeypatch):
    from bot import cloud_llm
    client = CloudClient("k")
    monkeypatch.setattr(cloud_llm.requests, "post",
                        lambda *a, **k: _Post(429, {"error": {"message": "rate limit"}}))
    with pytest.raises(CloudError, match="rate limit"):
        client.chat([{"role": "user", "content": "hi"}])
    assert issubclass(CloudError, OllamaError)  # the app falls back to simple mode on it

    def boom(*a, **k):
        raise cloud_llm.requests.ConnectionError("offline")
    monkeypatch.setattr(cloud_llm.requests, "post", boom)
    with pytest.raises(CloudError):
        client.chat([{"role": "user", "content": "hi"}])
    monkeypatch.setattr(cloud_llm.requests, "post", lambda *a, **k: _Post(200, {"text": " الجو في أسوان "}))
    assert client.transcribe(b"audio") == "الجو في أسوان"
