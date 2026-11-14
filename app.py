"""The website. Run with:  streamlit run app.py"""
import json
import os
import re
import sys
import types
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# The host keeps imported modules in memory between updates. When the project's files
# change, drop this project's modules so the new code is imported together; otherwise an
# updated app.py can meet an old copy of a module and fail to start.
_ROOT = Path(__file__).resolve().parent
_stamp = max(f.stat().st_mtime for f in [*_ROOT.glob("*.py"), *(_ROOT / "bot").glob("*.py")])
_seen = sys.modules.setdefault("_demo_code_stamp", types.ModuleType("_demo_code_stamp"))
if getattr(_seen, "value", None) != _stamp:
    for _name in [m for m in sys.modules if m in ("brand", "config", "bot") or m.startswith("bot.")]:
        del sys.modules[_name]
    _seen.value = _stamp

import brand  # noqa: E402
import config  # noqa: E402
from bot import db, evaluate, extras, planner, profile, tools, travel  # noqa: E402
from bot.agent import TourismAgent
from bot.cities import CITIES, canonical_city
from bot.cloud_llm import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_VISION_MODEL, CloudClient
from bot.faq import FAQ
from bot.i18n import t
from bot.llm import OllamaClient, OllamaError
from bot.rag import KnowledgeBase
from bot.simple import LOCAL_MATCH, SimpleAgent, format_booking, is_english
from bot.textutil import tokenize

st.set_page_config(page_title=f"{config.BOT_NAME} | مساعد السياحة في مصر",
                   page_icon=brand.ICON, layout="centered")

EXAMPLES = {
    "ar": ["أروح فين في إسكندرية في يومين؟", "احكيلي عن قلعة قايتباي", "الجو عامل إيه في أسوان؟",
           "عايز فندق في الأقصر بأقل من 2000 جنيه", "عندي شكوى وعايز أكلم موظف"],
    "en": ["Tell me about Karnak temple", "What is the weather in Aswan?",
           "What tours can I book in Luxor?", "Hotels in Alexandria under 3000"],
}
PAGES = ["chat", "plan", "tools", "stats", "about"]
WEB_TOOL = "search_web_knowledge"
MONTHS = {"ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس",
                 "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August",
                 "September", "October", "November", "December"]}


def secret(name: str) -> str:
    """A setting from the environment or from Streamlit secrets ('' when not set)."""
    value = os.getenv(name, "")
    if not value:
        try:
            value = str(st.secrets.get(name, ""))
        except Exception:  # no secrets file at all
            value = ""
    return value.strip()


def code_version() -> float:
    """Changes whenever the bot's code or data changes, so the cached bot below is
    rebuilt after an update instead of serving the old version."""
    files = [config.KNOWLEDGE_PATH, config.FAQ_PATH, config.EVAL_PATH]
    return max(_stamp, *(f.stat().st_mtime for f in files))


@st.cache_resource
def load_bot(version: float, cloud_key: str, cloud_model: str, cloud_url: str):
    local = OllamaClient(config.OLLAMA_HOST, config.CHAT_MODEL, config.EMBED_MODEL,
                         num_ctx=config.NUM_CTX, temperature=config.TEMPERATURE)
    db.init_db(config.DB_PATH)
    kb = KnowledgeBase(config.KNOWLEDGE_PATH, client=local, cache_path=config.EMB_CACHE_PATH)
    cloud = CloudClient(cloud_key, model=cloud_model, base_url=cloud_url) if cloud_key else None
    return {"local": local, "kb": kb, "faq": FAQ(config.FAQ_PATH), "cloud": cloud,
            "local_agent": TourismAgent(local, kb, config.DB_PATH),
            "cloud_agent": TourismAgent(cloud, kb, config.DB_PATH) if cloud else None,
            "simple_agent": SimpleAgent(kb, config.DB_PATH)}


@st.cache_data(ttl=86400, show_spinner=False)
def cached_image(wiki_title: str):
    return extras.place_image(wiki_title)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rates():
    return extras.fetch_rates()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prayers(lat: float, lon: float, day: str):
    return travel.prayer_times(lat, lon)


bot = load_bot(code_version(), secret("CLOUD_API_KEY") or secret("GROQ_API_KEY"),
               secret("CLOUD_MODEL") or DEFAULT_MODEL, secret("CLOUD_BASE_URL") or DEFAULT_BASE_URL)
kb, faq, cloud = bot["kb"], bot["faq"], bot["cloud"]
local_ready = bot["local"].is_up() and bot["local"].has_model(config.CHAT_MODEL)
mode = "cloud" if cloud else "local" if local_ready else "simple"
PLACES = [d for d in kb.docs if "lat" in d]

state = st.session_state
state.setdefault("lang", "ar")
state.setdefault("nav", "chat")
state.setdefault("messages", [])
state.setdefault("profile", {})
state.setdefault("trip", [])
ui = state.lang                      # interface language: ar, en or fr
lang = "ar" if ui == "ar" else "en"  # the bot's content (answers, place names) is Arabic or English
st.markdown(brand.css(rtl=ui == "ar"), unsafe_allow_html=True)


def city_name(code: str) -> str:
    return CITIES[code]["ar"] if lang == "ar" else code


def place_name(doc: dict) -> str:
    return doc["title_ar"] if lang == "ar" else doc["title_en"]


def ask(question: str) -> None:
    """Send a question to the chat (from a button on any page)."""
    state.pending = question
    state.nav = "chat"


def save_place(place: dict) -> None:
    if place["id"] not in [p["id"] for p in state.trip]:
        state.trip.append(place)


def drop_place(place_id: str) -> None:
    state.trip = [p for p in state.trip if p["id"] != place_id]


def show_map(points: list[dict]) -> None:
    if points:
        st.map(pd.DataFrame(points), latitude="lat", longitude="lon", height=260,
               use_container_width=True)


# ---- answering -------------------------------------------------------------
def media_for(question: str) -> dict:
    """A photo, map points and the place itself when the question is clearly about a known place."""
    hits = kb.search(question, k=1, prefer_city=canonical_city(question))
    if not hits or kb.match_strength(question, hits[0]["doc"]) < LOCAL_MATCH:
        return {}
    doc = hits[0]["doc"]
    by_id = {d["id"]: d for d in kb.docs}
    spots = [by_id[i] for i in doc.get("stops", []) if i in by_id] or [doc]
    points = [{"name": d["title_ar"], "lat": d["lat"], "lon": d["lon"]} for d in spots if "lat" in d]
    out = {"points": points}
    if "lat" in doc and not doc.get("stops"):
        out["place"] = {k: doc[k] for k in ("id", "title_ar", "title_en", "city", "lat", "lon")}
        if doc.get("wiki"):
            out["image"] = cached_image(doc["wiki"])
    return out


def follow_ups(message: dict, language: str) -> list[str]:
    """Three next questions that make sense after an answer about a place."""
    place = message.get("place")
    if not place:
        return []
    out = [f"فيه إيه قريب من {place['title_ar']}؟" if language == "ar"
           else f"What is near {place['title_en']}?"]
    if place["city"] in CITIES:
        city = CITIES[place["city"]]["ar"] if language == "ar" else place["city"]
        out += ([f"الجو عامل إيه في {city}؟", f"فنادق في {city}"] if language == "ar"
                else [f"What is the weather in {city}?", f"Hotels in {city}"])
    return out


def answer(history: list[dict], question: str) -> dict:
    learned = profile.update(state.profile, question)
    if learned and len(tokenize(question)) <= 4:  # the message only introduces the traveller
        en = is_english(question)
        text = (t("nice_to_meet", "en" if en else "ar", name=learned["name"]) if "name" in learned
                else t("kids_noted", "en" if en else "ar"))
        db.log_question(config.DB_PATH, question, "tool")
        return {"role": "assistant", "content": text, "question": question}

    stored = faq.match(question)  # a frequently asked question: answer it as written
    if stored:
        db.log_question(config.DB_PATH, question, "faq")
        return {"role": "assistant", "content": stored, "question": question}
    used_model = mode != "simple"
    about = profile.describe(state.profile)
    try:
        if mode == "cloud":
            result = bot["cloud_agent"].reply(history, question, about)
        elif mode == "local":
            result = bot["local_agent"].reply(history, question, about)
        else:
            raise OllamaError("no model")
    except OllamaError:
        used_model = False
        result = bot["simple_agent"].reply(history, question)

    worked = [e["tool"] for e in result.tool_events if "error" not in e["result"]]
    if not result.answered:
        source = "none"
    elif any(tool != WEB_TOOL for tool in worked):
        source = "tool"
    elif worked:
        source = "web"
    elif used_model:
        source = "model"
    else:
        source = "knowledge" if result.sources else "tool"
    db.log_question(config.DB_PATH, question, source)

    message = {"role": "assistant", "content": result.text, "question": question}
    if source in ("knowledge", "model"):
        message.update(media_for(question))
    return message


def speakable(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # keep link text, drop the address
    return re.sub(r"[*#`_>|]", " ", text)


def listen_button(text: str, key_lang: str) -> None:
    """Reads the reply aloud with the browser's own voice (no service involved)."""
    voice = "en-US" if is_english(text) else "ar-EG"
    components.html(f"""
<button id="b" style="font-family:Tahoma,Arial,sans-serif;font-size:13px;cursor:pointer;
  border:1.5px solid {brand.GOLD};border-radius:999px;background:rgba(255,255,255,.55);
  color:{brand.NIGHT};padding:5px 14px">{t('listen', key_lang)}</button>
<script>
const text = {json.dumps(speakable(text))};
document.getElementById("b").onclick = () => {{
  if (!("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "{voice}";
  window.speechSynthesis.speak(u);
}};
</script>""", height=42)


def whatsapp_link(text: str) -> str:
    return "https://wa.me/?text=" + quote(speakable(text)[:1500])


# ---- sidebar ---------------------------------------------------------------
with st.sidebar:
    state.lang = st.radio("Language", ["ar", "en", "fr"], horizontal=True, label_visibility="collapsed",
                          index=["ar", "en", "fr"].index(ui),
                          format_func={"ar": "العربية", "en": "English", "fr": "Français"}.get)
    if state.lang != ui:
        st.rerun()

    with st.expander(t("status", ui)):
        if mode == "cloud":
            st.success(f"AI: {cloud.chat_model}")
        elif mode == "local":
            st.success(f"AI: {config.CHAT_MODEL} (Ollama)")
        else:
            st.info("الوضع البسيط: بحث + أدوات، بدون موديل." if lang == "ar"
                    else "Simple mode: search and tools, no model.")
        st.caption(f"{len(kb.docs)} " + ("موضوع في قاعدة المعرفة" if lang == "ar"
                                         else "topics in the knowledge base"))

    st.header(t("faq", ui))
    for i, question in enumerate(faq.questions(lang=lang)):
        st.button(question, key=f"faq_{i}", use_container_width=True, on_click=ask, args=(question,))

    st.header(t("quick_booking", ui))
    kind = st.radio(t("kind", ui), ["hotel", "tour"], horizontal=True,
                    format_func=lambda v: t(v, ui))
    catalogue = db.query(config.DB_PATH,
                         f"SELECT * FROM {'hotels' if kind == 'hotel' else 'tours'} ORDER BY city, name")
    with st.form("booking_form", clear_on_submit=True):
        item = st.selectbox(t("choice", ui), catalogue,
                            format_func=lambda r: f"{r['city']} · {r['name']}")
        customer = st.text_input(t("customer", ui))
        day = st.date_input(t("date", ui), value=date.today(), min_value=date.today())
        nights = st.number_input(t("nights", ui), 1, 30, 1) if kind == "hotel" else 1
        people = st.number_input(t("people", ui), 1, 20, 2)
        if st.form_submit_button(t("book", ui), use_container_width=True):
            if not customer.strip():
                st.error(t("need_name", ui))
            else:
                booking = tools.create_booking(config.DB_PATH, kind, item["id"], customer,
                                               day.isoformat(), nights=nights, people=people)
                if "error" in booking:
                    st.error(booking["error"])
                else:
                    state.messages.append({"role": "assistant",
                                           "content": format_booking(booking, lang == "en")})
                    state.nav = "chat"
                    st.rerun()

    st.header(t("bookings", ui))
    bookings = db.list_bookings(config.DB_PATH)
    if bookings:
        st.dataframe(
            [{"#": b["ref"], t("choice", ui): b["item_name"], t("customer", ui): b["customer_name"],
              t("date", ui): b["start_date"], t("total", ui): b["total_egp"], "": b["status"]}
             for b in bookings], hide_index=True, use_container_width=True)
        chosen = st.selectbox(t("ticket_pick", ui), bookings,
                              format_func=lambda b: f"{b['ref']} · {b['customer_name']}")
        st.download_button(t("ticket", ui), extras.ticket_html(chosen, config.BOT_NAME),
                           file_name=f"{chosen['ref']}.html", mime="text/html",
                           use_container_width=True)
    else:
        st.caption(t("no_bookings", ui))

    handoffs = db.list_handoffs(config.DB_PATH)
    if handoffs:
        st.header(t("handoffs", ui))
        for h in handoffs:
            st.caption(f"#{h['id']} · {h['reason']}")

    if st.button(t("new_chat", ui), use_container_width=True):
        state.messages = []
        state.profile = {}
        st.rerun()

# ---- header and navigation ---------------------------------------------------
st.markdown(brand.hero_html(config.BOT_NAME, t("tagline", ui), latin="DEMO AI"),
            unsafe_allow_html=True)
if state.nav not in PAGES:
    state.nav = "chat"
page = st.radio("nav", PAGES, key="nav", horizontal=True, label_visibility="collapsed",
                format_func=lambda p: t(f"nav_{p}", ui))

# ---- chat ------------------------------------------------------------------
if page == "chat":
    with st.chat_message("assistant", avatar=brand.ICON):
        st.markdown(t("welcome", ui, name=config.BOT_NAME))

    last = len(state.messages) - 1
    for i, message in enumerate(state.messages):
        if message["role"] == "user":
            st.markdown(brand.user_bubble(message["content"]), unsafe_allow_html=True)
            continue
        with st.chat_message("assistant", avatar=brand.ICON):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"]["url"], use_container_width=True)
                st.caption(f"[{t('photo_credit', ui)}]({message['image']['page']})")
            show_map(message.get("points") or [])
        if not message.get("question"):
            continue
        left, middle, right = st.columns([1, 1, 1])
        with left:
            if hasattr(st, "feedback"):
                rating = st.feedback("thumbs", key=f"fb_{i}")
                if rating is not None and not message.get("rated"):
                    db.log_feedback(config.DB_PATH, message["question"], rating)
                    message["rated"] = True
        if i == last:
            with middle:
                listen_button(message["content"], ui)
            if hasattr(st, "link_button"):
                right.link_button(t("share_wa", ui), whatsapp_link(message["content"]),
                                  use_container_width=True)
        place = message.get("place")
        if place:
            saved = place["id"] in [p["id"] for p in state.trip]
            st.button(t("trip_saved" if saved else "trip_save", ui), key=f"save_{i}",
                      disabled=saved, on_click=save_place, args=(place,))
        if i == last:
            suggestions = follow_ups(message, lang)
            if suggestions:
                st.markdown(f'<div class="nb-hint">{t("next_hint", ui)}</div>', unsafe_allow_html=True)
                for n, suggestion in enumerate(suggestions):
                    st.button(suggestion, key=f"next_{i}_{n}", use_container_width=True,
                              on_click=ask, args=(suggestion,))

    if not state.messages:
        st.markdown(f'<div class="nb-hint">{t("faq_hint", ui)}</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for i, example in enumerate((faq.questions(2, lang) + EXAMPLES[lang])[:4]):
            columns[i % 2].button(example, key=f"start_{i}", use_container_width=True,
                                  on_click=ask, args=(example,))

    if hasattr(st, "audio_input"):
        with st.expander(t("voice", ui)):
            if cloud:
                audio = st.audio_input(t("voice", ui), label_visibility="collapsed")
                if audio is not None:
                    data = audio.getvalue()
                    marker = (len(data), hash(data[:2000]))
                    if state.get("last_audio") != marker:
                        state.last_audio = marker
                        try:
                            heard = cloud.transcribe(data, audio.name or "voice.wav")
                        except OllamaError as error:
                            heard = ""
                            st.error(str(error))
                        if heard:
                            state.pending = heard
            else:
                st.caption(t("voice_off", ui))

    with st.expander(t("photo_ask", ui)):
        if cloud:
            photo = st.file_uploader(t("photo_label", ui), type=["jpg", "jpeg", "png", "webp"])
            if photo is not None:
                data = photo.getvalue()
                marker = (len(data), hash(data[:2000]))
                if state.get("last_photo") != marker:
                    state.last_photo = marker
                    prompt_text = ("What monument or place in Egypt is in this photo? Give its name and two "
                                   "or three facts for a tourist. If it shows people rather than a place, do "
                                   "not identify them; say you can only help with places. If you are not "
                                   "sure, say so. " + ("Answer in English." if lang == "en"
                                                       else "Answer in Egyptian Arabic."))
                    try:
                        seen = cloud.describe_image(data, photo.type or "image/jpeg", prompt_text,
                                                    secret("CLOUD_VISION_MODEL") or DEFAULT_VISION_MODEL)
                    except OllamaError as error:
                        seen = ""
                        st.error(str(error))
                    if seen:
                        state.messages += [{"role": "user", "content": t("photo_sent", ui)},
                                           {"role": "assistant", "content": seen}]
                        db.log_question(config.DB_PATH, "(photo)", "model")
                        st.rerun()
        else:
            st.caption(t("photo_off", ui))

    prompt = st.chat_input(t("ask", ui)) or state.pop("pending", None)
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in state.messages]
        state.messages.append({"role": "user", "content": prompt})
        st.markdown(brand.user_bubble(prompt), unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=brand.ICON):
            with st.spinner(t("thinking", ui)):
                reply = answer(history, prompt)
        state.messages.append(reply)
        st.rerun()

# ---- trip: planner, near me, my trip -----------------------------------------
elif page == "plan":
    city = st.selectbox(t("city", ui), list(CITIES), format_func=city_name)
    days = st.slider(t("days", ui), 1, 7, 2)
    interests = st.multiselect(t("interests", ui), list(planner.INTERESTS), default=["history"],
                               format_func=lambda k: planner.INTERESTS[k][lang])
    kids = st.checkbox(t("with_kids", ui), value=bool(state.profile.get("kids")))
    if st.button(t("make_plan", ui), use_container_width=True):
        wanted = list(dict.fromkeys(interests + (["nature", "culture"] if kids else [])))
        state.plan = planner.build_plan(kb.docs, config.DB_PATH, city, days, wanted)
    plan = state.get("plan")
    if plan:
        if not plan["enough"]:
            st.info(t("plan_short", ui))
        for number, day in enumerate(plan["days"], 1):
            rows = "".join(
                f"<p><span>{planner.SLOTS[lang][item['slot']]}:</span> "
                f"<b>{item['title'] if lang == 'ar' else item['title_en']}</b><br>{item['note']}</p>"
                for item in day) or f"<p>{t('free_day', ui)}</p>"
            st.markdown(f'<div class="nb-day"><h4>{t("day", ui)} {number}</h4>{rows}</div>',
                        unsafe_allow_html=True)
        show_map(plan["points"])

    st.subheader(t("near_title", ui))
    here = st.selectbox(t("near_where", ui), PLACES, format_func=place_name)
    close = travel.nearest(kb.docs, here["lat"], here["lon"], 6, exclude_id=here["id"])
    st.markdown("| | |\n|---|---|\n" + "\n".join(
        f"| {place_name(c['doc'])} | {c['km']:g} {t('km', ui)} |" for c in close))
    show_map([{"lat": here["lat"], "lon": here["lon"]}]
             + [{"lat": c["doc"]["lat"], "lon": c["doc"]["lon"]} for c in close])

    st.subheader(t("trip_title", ui))
    if state.trip:
        for saved_place in state.trip:
            left, right = st.columns([3, 1])
            left.markdown(f"**{place_name(saved_place)}** · {saved_place['city']}")
            right.button(t("trip_remove", ui), key=f"drop_{saved_place['id']}",
                         on_click=drop_place, args=(saved_place["id"],))
        show_map([{"lat": p["lat"], "lon": p["lon"]} for p in state.trip])
        listing = "\n".join(f"- {place_name(p)} ({p['city']})" for p in state.trip)
        summary = f"{t('trip_title', ui)} | {config.BOT_NAME}\n{listing}"
        left, right = st.columns(2)
        left.download_button(t("trip_download", ui), summary, file_name="my-trip.txt",
                             use_container_width=True)
        if hasattr(st, "link_button"):
            right.link_button(t("share_wa", ui), whatsapp_link(summary), use_container_width=True)
    else:
        st.caption(t("trip_empty", ui))

# ---- travel tools ------------------------------------------------------------
elif page == "tools":
    tabs = st.tabs([t(f"tab_{name}", ui) for name in
                    ("budget", "money", "route", "pack", "days", "phrases")])

    with tabs[0]:
        st.caption(t("budget_intro", ui))
        city = st.selectbox(t("city", ui), list(CITIES), format_func=city_name, key="budget_city")
        hotels = db.query(config.DB_PATH,
                          "SELECT * FROM hotels WHERE city = ? ORDER BY price_per_night_egp", (city,))
        city_tours = db.query(config.DB_PATH,
                              "SELECT * FROM tours WHERE city = ? ORDER BY price_per_person_egp", (city,))
        hotel = st.selectbox(t("hotel", ui), [None] + hotels,
                             format_func=lambda h: t("no_hotel", ui) if h is None
                             else f"{h['name']} · {h['price_per_night_egp']:,} {t('egp', ui)}")
        left, right = st.columns(2)
        nights = left.number_input(t("nights", ui), 1, 30, 3, key="budget_nights")
        people = right.number_input(t("people", ui), 1, 20, 2, key="budget_people")
        picked = st.multiselect(t("tours", ui), city_tours,
                                format_func=lambda x: f"{x['name']} · {x['price_per_person_egp']:,} {t('egp', ui)}")
        budget = extras.trip_budget(config.DB_PATH, hotel["id"] if hotel else None, nights, people,
                                    [x["id"] for x in picked])
        if budget["lines"]:
            st.markdown("| | | |\n|---|---|---|\n" + "\n".join(
                f"| {line['item']} | {line['detail']} | {line['cost']:,} {t('egp', ui)} |"
                for line in budget["lines"]))
        left, right = st.columns(2)
        left.metric(t("total", ui), f"{budget['total_egp']:,} {t('egp', ui)}")
        right.metric(t("per_person", ui), f"{budget['per_person_egp']:,} {t('egp', ui)}")

    with tabs[1]:
        rates = cached_rates()
        if "error" in rates:
            st.warning(t("rates_off", ui))
        else:
            codes = list(rates["rates"])
            amount = st.number_input(t("amount", ui), min_value=0.0, value=100.0, step=10.0)
            left, right = st.columns(2)
            source = left.selectbox(t("from", ui), codes, index=codes.index("USD"))
            target = right.selectbox(t("to", ui), codes, index=codes.index("EGP"))
            value = extras.convert(amount, source, target, rates["rates"])
            st.metric(f"{amount:,.2f} {source}", f"{value:,.2f} {target}")
            st.markdown("| | |\n|---|---|\n" + "\n".join(
                f"| 1 {code} | {extras.convert(1, code, 'EGP', rates['rates']):,.2f} {t('egp', ui)} |"
                for code in codes if code != "EGP"))
            st.caption(t("rates_note", ui, when=rates["updated"]))

    with tabs[2]:
        left, right = st.columns(2)
        start = left.selectbox(t("route_from", ui), list(CITIES), format_func=city_name, key="route_a")
        end = right.selectbox(t("route_to", ui), list(CITIES), index=2, format_func=city_name,
                              key="route_b")
        options = travel.route(start, end)
        if options:
            st.markdown(f"| {t('mode', ui)} | {t('duration', ui)} | {t('note', ui)} |\n|---|---|---|\n"
                        + "\n".join(f"| {travel.MODES[o['mode']][lang]} | {o['duration']} | {o['note']} |"
                                    for o in options))
            st.caption(t("route_note", ui))
        elif start != end:
            st.info(t("route_none", ui))
        show_map([{"lat": CITIES[c]["lat"], "lon": CITIES[c]["lon"]} for c in {start, end}])

    with tabs[3]:
        left, right = st.columns(2)
        pack_city = left.selectbox(t("city", ui), list(CITIES), format_func=city_name, key="pack_city")
        month = right.selectbox(t("month", ui), list(range(1, 13)), index=date.today().month - 1,
                                format_func=lambda m: MONTHS[lang][m - 1])
        st.markdown("\n".join(f"- {entry}" for entry in travel.packing_list(pack_city, month, lang)))

    with tabs[4]:
        pray_city = st.selectbox(t("city", ui), list(CITIES), format_func=city_name, key="pray_city")
        st.subheader(t("prayer_today", ui))
        prayers = cached_prayers(CITIES[pray_city]["lat"], CITIES[pray_city]["lon"],
                                 date.today().isoformat())
        if "error" in prayers:
            st.warning(t("prayer_off", ui))
        else:
            st.markdown("| | |\n|---|---|\n" + "\n".join(
                f"| {ar if lang == 'ar' else key} | {prayers['times'][key]} |"
                for key, ar in travel.PRAYERS))
        st.subheader(t("holidays", ui))
        st.markdown(f"| {t('holiday', ui)} | {t('date', ui)} | {t('days_left', ui)} |\n|---|---|---|\n"
                    + "\n".join(f"| {h['name']} | {h['date']} | {h['days_left']} |"
                                for h in travel.upcoming_holidays(date.today(), lang)))
        st.caption(t("holidays_moving", ui)
                   + "، ".join(ar if lang == "ar" else en for ar, en in travel.MOVING_HOLIDAYS) + ". "
                   + t("holidays_note", ui))

    with tabs[5]:
        st.markdown(f"| {t('phrase_ar', ui)} | {t('phrase_say', ui)} | {t('phrase_en', ui)} |\n"
                    "|---|---|---|\n" + "\n".join(f"| {ar} | {say} | {en} |"
                                                  for ar, say, en in extras.PHRASES))

# ---- statistics --------------------------------------------------------------
elif page == "stats":
    numbers = db.stats(config.DB_PATH)
    survey = db.survey_summary(config.DB_PATH)
    cols = st.columns(4)
    cols[0].metric(t("st_questions", ui), numbers["questions"])
    cols[1].metric(t("st_unanswered", ui), numbers["unanswered"])
    cols[2].metric(t("st_bookings", ui), numbers["bookings"])
    cols[3].metric(t("st_likes", ui), f"{numbers['likes']} / {numbers['dislikes']}")
    if numbers["by_source"]:
        st.subheader(t("st_by_source", ui))
        st.bar_chart(pd.DataFrame(
            {"n": list(numbers["by_source"].values())},
            index=[t(f"src_{source}", ui) for source in numbers["by_source"]]), color=brand.GOLD)
    else:
        st.caption(t("st_empty", ui))

    st.subheader(t("survey_results", ui))
    cols = st.columns(4)
    cols[0].metric(t("survey_n", ui), survey["responses"])
    for column, key in zip(cols[1:], ("ease", "useful", "design")):
        column.metric(t(f"survey_{key}", ui), f"{survey[key]} / 5" if survey[key] else "-")

    # Texts written by other visitors need the admin password.
    admin = secret("ADMIN_PASSWORD")
    if admin and st.text_input(t("st_password", ui), type="password") == admin:
        for title, rows in ((t("st_top", ui), numbers["top_questions"]),
                            (t("st_missing", ui), numbers["unanswered_list"]),
                            (t("st_disliked", ui), numbers["disliked_list"]),
                            (t("survey_comments", ui), [{"text": c} for c in survey["comments"]])):
            st.subheader(title)
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.caption(t("st_empty", ui))
    else:
        st.caption(t("st_locked", ui))

# ---- about -------------------------------------------------------------------
else:
    st.subheader(t("why_name", ui))
    st.markdown(t("why_name_text", ui))

    st.subheader(t("about_how", ui))
    steps = ([("السؤال", "السائح يكتب أو يتكلم بالعربية أو الإنجليزية."),
              ("الأسئلة الشائعة", "لو السؤال من الأسئلة الشائعة، الإجابة المكتوبة ترجع فورًا."),
              ("قاعدة المعرفة", f"البحث في {len(kb.docs)} موضوعًا عن مصر عن أقرب موضوع للسؤال."),
              ("الأدوات", "الطقس والفنادق والجولات والحجز تأتي من أدوات حقيقية، لا من التأليف."),
              ("المصادر الخارجية", "لو الموضوع غير موجود، البحث في ويكيبيديا مع ذكر المصدر."),
              ("الرد", "إجابة مباشرة، ومعها صورة وخريطة للمكان.")] if lang == "ar" else
             [("Question", "The traveller types or speaks in Arabic or English."),
              ("FAQ", "A frequently asked question gets its written answer at once."),
              ("Knowledge base", f"A search across {len(kb.docs)} topics about Egypt for the closest one."),
              ("Tools", "Weather, hotels, tours and bookings come from real tools, not invention."),
              ("External sources", "If the topic is missing, Wikipedia is searched and credited."),
              ("Answer", "A direct answer, with a photo and a map of the place.")])
    st.markdown("".join(
        f'<div class="nb-day"><h4>{number}. {title}</h4><p>{text}</p></div>'
        for number, (title, text) in enumerate(steps, 1)), unsafe_allow_html=True)

    st.subheader(t("about_tools", ui))
    st.markdown("- Python, Streamlit\n- Ollama + Qwen2.5 (" + ("موديل محلي" if lang == "ar" else "local model")
                + "), Groq (" + ("موديل سحابي اختياري" if lang == "ar" else "optional cloud model") + ")\n"
                "- RAG: bge-m3 embeddings, BM25\n- SQLite\n- Wikipedia, Wikivoyage, Open-Meteo, wttr.in, "
                "open.er-api.com, aladhan.com")

    st.subheader(t("about_limits", ui))
    st.markdown("- الفنادق والجولات والأسعار بيانات تجريبية، ولا يوجد حجز أو دفع حقيقي.\n"
                "- بدون موديل، البوت يبحث ويعرض ولا يستنتج.\n"
                "- أسعار التذاكر ومواعيد المزارات غير مذكورة لأنها تتغير؛ راجع المصدر الرسمي.\n"
                "- الإحصائيات والحجوزات تُمسح عند إعادة تشغيل الموقع."
                if lang == "ar" else
                "- Hotels, tours and prices are demo data; there is no real booking or payment.\n"
                "- Without a model the bot searches and shows, it does not reason.\n"
                "- Ticket prices and opening hours are left out because they change; check the official source.\n"
                "- Statistics and bookings are cleared when the site restarts.")

    st.subheader(t("eval_title", ui))
    total_cases = len(json.loads(config.EVAL_PATH.read_text(encoding="utf-8")))
    st.caption(t("eval_text", ui, n=total_cases))
    if st.button(t("eval_run", ui), use_container_width=True):
        state.eval = evaluate.run(kb, faq, config.EVAL_PATH)
    report = state.get("eval")
    if report:
        cols = st.columns(4)
        cols[0].metric(t("eval_accuracy", ui), f"{report['accuracy']}%")
        for column, kind_key in zip(cols[1:], ("topic", "faq", "tool")):
            part = report["by_type"].get(kind_key, {"passed": 0, "total": 0})
            column.metric(t(f"eval_{kind_key}", ui), f"{part['passed']} / {part['total']}")
        if report["failures"]:
            st.subheader(t("eval_failed", ui))
            st.dataframe([{k: row[k] for k in ("question", "expected", "got")}
                          for row in report["failures"]], hide_index=True, use_container_width=True)

    st.subheader(t("survey_title", ui))
    if state.get("survey_done"):
        st.success(t("survey_thanks", ui))
    else:
        with st.form("survey_form"):
            ease = st.slider(t("survey_ease", ui), 1, 5, 3)
            useful = st.slider(t("survey_useful", ui), 1, 5, 3)
            design = st.slider(t("survey_design", ui), 1, 5, 3)
            comment = st.text_area(t("survey_comment", ui))
            if st.form_submit_button(t("survey_send", ui), use_container_width=True):
                db.log_survey(config.DB_PATH, ease, useful, design, comment)
                state.survey_done = True
                st.rerun()
