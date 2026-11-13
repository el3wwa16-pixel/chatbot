"""The website. Run with:  streamlit run app.py"""
import json
import os
import re
from datetime import date
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import brand
import config
from bot import db, evaluate, extras, planner, profile, tools, travel
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
st.markdown(brand.css(), unsafe_allow_html=True)

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
    files = list((config.BASE_DIR / "bot").glob("*.py")) + [config.KNOWLEDGE_PATH, config.FAQ_PATH,
                                                           config.EVAL_PATH]
    return max(f.stat().st_mtime for f in files)


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
lang = state.lang


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
    state.lang = st.radio("Language", ["ar", "en"], horizontal=True, label_visibility="collapsed",
                          index=["ar", "en"].index(lang),
                          format_func=lambda v: "العربية" if v == "ar" else "English")
    if state.lang != lang:
        st.rerun()

    with st.expander(t("status", lang)):
        if mode == "cloud":
            st.success(f"AI: {cloud.chat_model}")
        elif mode == "local":
            st.success(f"AI: {config.CHAT_MODEL} (Ollama)")
        else:
            st.info("الوضع البسيط: بحث + أدوات، بدون موديل." if lang == "ar"
                    else "Simple mode: search and tools, no model.")
        st.caption(f"{len(kb.docs)} " + ("موضوع في قاعدة المعرفة" if lang == "ar"
                                         else "topics in the knowledge base"))

    st.header(t("faq", lang))
    for i, question in enumerate(faq.questions(lang=lang)):
        st.button(question, key=f"faq_{i}", use_container_width=True, on_click=ask, args=(question,))

    st.header(t("quick_booking", lang))
    kind = st.radio(t("kind", lang), ["hotel", "tour"], horizontal=True,
                    format_func=lambda v: t(v, lang))
    catalogue = db.query(config.DB_PATH,
                         f"SELECT * FROM {'hotels' if kind == 'hotel' else 'tours'} ORDER BY city, name")
    with st.form("booking_form", clear_on_submit=True):
        item = st.selectbox(t("choice", lang), catalogue,
                            format_func=lambda r: f"{r['city']} · {r['name']}")
        customer = st.text_input(t("customer", lang))
        day = st.date_input(t("date", lang), value=date.today(), min_value=date.today())
        nights = st.number_input(t("nights", lang), 1, 30, 1) if kind == "hotel" else 1
        people = st.number_input(t("people", lang), 1, 20, 2)
        if st.form_submit_button(t("book", lang), use_container_width=True):
            if not customer.strip():
                st.error(t("need_name", lang))
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

    st.header(t("bookings", lang))
    bookings = db.list_bookings(config.DB_PATH)
    if bookings:
        st.dataframe(
            [{"#": b["ref"], t("choice", lang): b["item_name"], t("customer", lang): b["customer_name"],
              t("date", lang): b["start_date"], t("total", lang): b["total_egp"], "": b["status"]}
             for b in bookings], hide_index=True, use_container_width=True)
        chosen = st.selectbox(t("ticket_pick", lang), bookings,
                              format_func=lambda b: f"{b['ref']} · {b['customer_name']}")
        st.download_button(t("ticket", lang), extras.ticket_html(chosen, config.BOT_NAME),
                           file_name=f"{chosen['ref']}.html", mime="text/html",
                           use_container_width=True)
    else:
        st.caption(t("no_bookings", lang))

    handoffs = db.list_handoffs(config.DB_PATH)
    if handoffs:
        st.header(t("handoffs", lang))
        for h in handoffs:
            st.caption(f"#{h['id']} · {h['reason']}")

    if st.button(t("new_chat", lang), use_container_width=True):
        state.messages = []
        state.profile = {}
        st.rerun()

# ---- header and navigation ---------------------------------------------------
st.markdown(brand.hero_html(config.BOT_NAME, t("tagline", lang), latin="DEMO AI"),
            unsafe_allow_html=True)
if state.nav not in PAGES:
    state.nav = "chat"
page = st.radio("nav", PAGES, key="nav", horizontal=True, label_visibility="collapsed",
                format_func=lambda p: t(f"nav_{p}", lang))

# ---- chat ------------------------------------------------------------------
if page == "chat":
    with st.chat_message("assistant", avatar=brand.ICON):
        st.markdown(t("welcome", lang, name=config.BOT_NAME))

    last = len(state.messages) - 1
    for i, message in enumerate(state.messages):
        if message["role"] == "user":
            st.markdown(brand.user_bubble(message["content"]), unsafe_allow_html=True)
            continue
        with st.chat_message("assistant", avatar=brand.ICON):
            st.markdown(message["content"])
            if message.get("image"):
                st.image(message["image"]["url"], use_container_width=True)
                st.caption(f"[{t('photo_credit', lang)}]({message['image']['page']})")
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
                listen_button(message["content"], lang)
            if hasattr(st, "link_button"):
                right.link_button(t("share_wa", lang), whatsapp_link(message["content"]),
                                  use_container_width=True)
        place = message.get("place")
        if place:
            saved = place["id"] in [p["id"] for p in state.trip]
            st.button(t("trip_saved" if saved else "trip_save", lang), key=f"save_{i}",
                      disabled=saved, on_click=save_place, args=(place,))
        if i == last:
            suggestions = follow_ups(message, lang)
            if suggestions:
                st.markdown(f'<div class="nb-hint">{t("next_hint", lang)}</div>', unsafe_allow_html=True)
                for n, suggestion in enumerate(suggestions):
                    st.button(suggestion, key=f"next_{i}_{n}", use_container_width=True,
                              on_click=ask, args=(suggestion,))

    if not state.messages:
        st.markdown(f'<div class="nb-hint">{t("faq_hint", lang)}</div>', unsafe_allow_html=True)
        columns = st.columns(2)
        for i, example in enumerate((faq.questions(2, lang) + EXAMPLES[lang])[:4]):
            columns[i % 2].button(example, key=f"start_{i}", use_container_width=True,
                                  on_click=ask, args=(example,))

    if hasattr(st, "audio_input"):
        with st.expander(t("voice", lang)):
            if cloud:
                audio = st.audio_input(t("voice", lang), label_visibility="collapsed")
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
                st.caption(t("voice_off", lang))

    with st.expander(t("photo_ask", lang)):
        if cloud:
            photo = st.file_uploader(t("photo_label", lang), type=["jpg", "jpeg", "png", "webp"])
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
                        state.messages += [{"role": "user", "content": t("photo_sent", lang)},
                                           {"role": "assistant", "content": seen}]
                        db.log_question(config.DB_PATH, "(photo)", "model")
                        st.rerun()
        else:
            st.caption(t("photo_off", lang))

    prompt = st.chat_input(t("ask", lang)) or state.pop("pending", None)
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in state.messages]
        state.messages.append({"role": "user", "content": prompt})
        st.markdown(brand.user_bubble(prompt), unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=brand.ICON):
            with st.spinner(t("thinking", lang)):
                reply = answer(history, prompt)
        state.messages.append(reply)
        st.rerun()

# ---- trip: planner, near me, my trip -----------------------------------------
elif page == "plan":
    city = st.selectbox(t("city", lang), list(CITIES), format_func=city_name)
    days = st.slider(t("days", lang), 1, 7, 2)
    interests = st.multiselect(t("interests", lang), list(planner.INTERESTS), default=["history"],
                               format_func=lambda k: planner.INTERESTS[k][lang])
    kids = st.checkbox(t("with_kids", lang), value=bool(state.profile.get("kids")))
    if st.button(t("make_plan", lang), use_container_width=True):
        wanted = list(dict.fromkeys(interests + (["nature", "culture"] if kids else [])))
        state.plan = planner.build_plan(kb.docs, config.DB_PATH, city, days, wanted)
    plan = state.get("plan")
    if plan:
        if not plan["enough"]:
            st.info(t("plan_short", lang))
        for number, day in enumerate(plan["days"], 1):
            rows = "".join(
                f"<p><span>{planner.SLOTS[lang][item['slot']]}:</span> "
                f"<b>{item['title'] if lang == 'ar' else item['title_en']}</b><br>{item['note']}</p>"
                for item in day) or f"<p>{t('free_day', lang)}</p>"
            st.markdown(f'<div class="nb-day"><h4>{t("day", lang)} {number}</h4>{rows}</div>',
                        unsafe_allow_html=True)
        show_map(plan["points"])

    st.subheader(t("near_title", lang))
    here = st.selectbox(t("near_where", lang), PLACES, format_func=place_name)
    close = travel.nearest(kb.docs, here["lat"], here["lon"], 6, exclude_id=here["id"])
    st.markdown("| | |\n|---|---|\n" + "\n".join(
        f"| {place_name(c['doc'])} | {c['km']:g} {t('km', lang)} |" for c in close))
    show_map([{"lat": here["lat"], "lon": here["lon"]}]
             + [{"lat": c["doc"]["lat"], "lon": c["doc"]["lon"]} for c in close])

    st.subheader(t("trip_title", lang))
    if state.trip:
        for saved_place in state.trip:
            left, right = st.columns([3, 1])
            left.markdown(f"**{place_name(saved_place)}** · {saved_place['city']}")
            right.button(t("trip_remove", lang), key=f"drop_{saved_place['id']}",
                         on_click=drop_place, args=(saved_place["id"],))
        show_map([{"lat": p["lat"], "lon": p["lon"]} for p in state.trip])
        listing = "\n".join(f"- {place_name(p)} ({p['city']})" for p in state.trip)
        summary = f"{t('trip_title', lang)} | {config.BOT_NAME}\n{listing}"
        left, right = st.columns(2)
        left.download_button(t("trip_download", lang), summary, file_name="my-trip.txt",
                             use_container_width=True)
        if hasattr(st, "link_button"):
            right.link_button(t("share_wa", lang), whatsapp_link(summary), use_container_width=True)
    else:
        st.caption(t("trip_empty", lang))

# ---- travel tools ------------------------------------------------------------
elif page == "tools":
    tabs = st.tabs([t(f"tab_{name}", lang) for name in
                    ("budget", "money", "route", "pack", "days", "phrases")])

    with tabs[0]:
        st.caption(t("budget_intro", lang))
        city = st.selectbox(t("city", lang), list(CITIES), format_func=city_name, key="budget_city")
        hotels = db.query(config.DB_PATH,
                          "SELECT * FROM hotels WHERE city = ? ORDER BY price_per_night_egp", (city,))
        city_tours = db.query(config.DB_PATH,
                              "SELECT * FROM tours WHERE city = ? ORDER BY price_per_person_egp", (city,))
        hotel = st.selectbox(t("hotel", lang), [None] + hotels,
                             format_func=lambda h: t("no_hotel", lang) if h is None
                             else f"{h['name']} · {h['price_per_night_egp']:,} {t('egp', lang)}")
        left, right = st.columns(2)
        nights = left.number_input(t("nights", lang), 1, 30, 3, key="budget_nights")
        people = right.number_input(t("people", lang), 1, 20, 2, key="budget_people")
        picked = st.multiselect(t("tours", lang), city_tours,
                                format_func=lambda x: f"{x['name']} · {x['price_per_person_egp']:,} {t('egp', lang)}")
        budget = extras.trip_budget(config.DB_PATH, hotel["id"] if hotel else None, nights, people,
                                    [x["id"] for x in picked])
        if budget["lines"]:
            st.markdown("| | | |\n|---|---|---|\n" + "\n".join(
                f"| {line['item']} | {line['detail']} | {line['cost']:,} {t('egp', lang)} |"
                for line in budget["lines"]))
        left, right = st.columns(2)
        left.metric(t("total", lang), f"{budget['total_egp']:,} {t('egp', lang)}")
        right.metric(t("per_person", lang), f"{budget['per_person_egp']:,} {t('egp', lang)}")

    with tabs[1]:
        rates = cached_rates()
        if "error" in rates:
            st.warning(t("rates_off", lang))
        else:
            codes = list(rates["rates"])
            amount = st.number_input(t("amount", lang), min_value=0.0, value=100.0, step=10.0)
            left, right = st.columns(2)
            source = left.selectbox(t("from", lang), codes, index=codes.index("USD"))
            target = right.selectbox(t("to", lang), codes, index=codes.index("EGP"))
            value = extras.convert(amount, source, target, rates["rates"])
            st.metric(f"{amount:,.2f} {source}", f"{value:,.2f} {target}")
            st.markdown("| | |\n|---|---|\n" + "\n".join(
                f"| 1 {code} | {extras.convert(1, code, 'EGP', rates['rates']):,.2f} {t('egp', lang)} |"
                for code in codes if code != "EGP"))
            st.caption(t("rates_note", lang, when=rates["updated"]))

    with tabs[2]:
        left, right = st.columns(2)
        start = left.selectbox(t("route_from", lang), list(CITIES), format_func=city_name, key="route_a")
        end = right.selectbox(t("route_to", lang), list(CITIES), index=2, format_func=city_name,
                              key="route_b")
        options = travel.route(start, end)
        if options:
            st.markdown(f"| {t('mode', lang)} | {t('duration', lang)} | {t('note', lang)} |\n|---|---|---|\n"
                        + "\n".join(f"| {travel.MODES[o['mode']][lang]} | {o['duration']} | {o['note']} |"
                                    for o in options))
            st.caption(t("route_note", lang))
        elif start != end:
            st.info(t("route_none", lang))
        show_map([{"lat": CITIES[c]["lat"], "lon": CITIES[c]["lon"]} for c in {start, end}])

    with tabs[3]:
        left, right = st.columns(2)
        pack_city = left.selectbox(t("city", lang), list(CITIES), format_func=city_name, key="pack_city")
        month = right.selectbox(t("month", lang), list(range(1, 13)), index=date.today().month - 1,
                                format_func=lambda m: MONTHS[lang][m - 1])
        st.markdown("\n".join(f"- {entry}" for entry in travel.packing_list(pack_city, month, lang)))

    with tabs[4]:
        pray_city = st.selectbox(t("city", lang), list(CITIES), format_func=city_name, key="pray_city")
        st.subheader(t("prayer_today", lang))
        prayers = cached_prayers(CITIES[pray_city]["lat"], CITIES[pray_city]["lon"],
                                 date.today().isoformat())
        if "error" in prayers:
            st.warning(t("prayer_off", lang))
        else:
            st.markdown("| | |\n|---|---|\n" + "\n".join(
                f"| {ar if lang == 'ar' else key} | {prayers['times'][key]} |"
                for key, ar in travel.PRAYERS))
        st.subheader(t("holidays", lang))
        st.markdown(f"| {t('holiday', lang)} | {t('date', lang)} | {t('days_left', lang)} |\n|---|---|---|\n"
                    + "\n".join(f"| {h['name']} | {h['date']} | {h['days_left']} |"
                                for h in travel.upcoming_holidays(date.today(), lang)))
        st.caption(t("holidays_moving", lang)
                   + "، ".join(ar if lang == "ar" else en for ar, en in travel.MOVING_HOLIDAYS) + ". "
                   + t("holidays_note", lang))

    with tabs[5]:
        st.markdown(f"| {t('phrase_ar', lang)} | {t('phrase_say', lang)} | {t('phrase_en', lang)} |\n"
                    "|---|---|---|\n" + "\n".join(f"| {ar} | {say} | {en} |"
                                                  for ar, say, en in extras.PHRASES))

# ---- statistics --------------------------------------------------------------
elif page == "stats":
    numbers = db.stats(config.DB_PATH)
    survey = db.survey_summary(config.DB_PATH)
    cols = st.columns(4)
    cols[0].metric(t("st_questions", lang), numbers["questions"])
    cols[1].metric(t("st_unanswered", lang), numbers["unanswered"])
    cols[2].metric(t("st_bookings", lang), numbers["bookings"])
    cols[3].metric(t("st_likes", lang), f"{numbers['likes']} / {numbers['dislikes']}")
    if numbers["by_source"]:
        st.subheader(t("st_by_source", lang))
        st.bar_chart(pd.DataFrame(
            {"n": list(numbers["by_source"].values())},
            index=[t(f"src_{source}", lang) for source in numbers["by_source"]]), color=brand.GOLD)
    else:
        st.caption(t("st_empty", lang))

    st.subheader(t("survey_results", lang))
    cols = st.columns(4)
    cols[0].metric(t("survey_n", lang), survey["responses"])
    for column, key in zip(cols[1:], ("ease", "useful", "design")):
        column.metric(t(f"survey_{key}", lang), f"{survey[key]} / 5" if survey[key] else "-")

    # Texts written by other visitors need the admin password.
    admin = secret("ADMIN_PASSWORD")
    if admin and st.text_input(t("st_password", lang), type="password") == admin:
        for title, rows in ((t("st_top", lang), numbers["top_questions"]),
                            (t("st_missing", lang), numbers["unanswered_list"]),
                            (t("st_disliked", lang), numbers["disliked_list"]),
                            (t("survey_comments", lang), [{"text": c} for c in survey["comments"]])):
            st.subheader(title)
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.caption(t("st_empty", lang))
    else:
        st.caption(t("st_locked", lang))

# ---- about -------------------------------------------------------------------
else:
    st.subheader(t("why_name", lang))
    st.markdown(t("why_name_text", lang))

    st.subheader(t("about_how", lang))
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

    st.subheader(t("about_tools", lang))
    st.markdown("- Python, Streamlit\n- Ollama + Qwen2.5 (" + ("موديل محلي" if lang == "ar" else "local model")
                + "), Groq (" + ("موديل سحابي اختياري" if lang == "ar" else "optional cloud model") + ")\n"
                "- RAG: bge-m3 embeddings, BM25\n- SQLite\n- Wikipedia, Wikivoyage, Open-Meteo, wttr.in, "
                "open.er-api.com, aladhan.com")

    st.subheader(t("about_limits", lang))
    st.markdown("- الفنادق والجولات والأسعار بيانات تجريبية، ولا يوجد حجز أو دفع حقيقي.\n"
                "- بدون موديل، البوت يبحث ويعرض ولا يستنتج.\n"
                "- أسعار التذاكر ومواعيد المزارات غير مذكورة لأنها تتغير؛ راجع المصدر الرسمي.\n"
                "- الإحصائيات والحجوزات تُمسح عند إعادة تشغيل الموقع."
                if lang == "ar" else
                "- Hotels, tours and prices are demo data; there is no real booking or payment.\n"
                "- Without a model the bot searches and shows, it does not reason.\n"
                "- Ticket prices and opening hours are left out because they change; check the official source.\n"
                "- Statistics and bookings are cleared when the site restarts.")

    st.subheader(t("eval_title", lang))
    total_cases = len(json.loads(config.EVAL_PATH.read_text(encoding="utf-8")))
    st.caption(t("eval_text", lang, n=total_cases))
    if st.button(t("eval_run", lang), use_container_width=True):
        state.eval = evaluate.run(kb, faq, config.EVAL_PATH)
    report = state.get("eval")
    if report:
        cols = st.columns(4)
        cols[0].metric(t("eval_accuracy", lang), f"{report['accuracy']}%")
        for column, kind_key in zip(cols[1:], ("topic", "faq", "tool")):
            part = report["by_type"].get(kind_key, {"passed": 0, "total": 0})
            column.metric(t(f"eval_{kind_key}", lang), f"{part['passed']} / {part['total']}")
        if report["failures"]:
            st.subheader(t("eval_failed", lang))
            st.dataframe([{k: row[k] for k in ("question", "expected", "got")}
                          for row in report["failures"]], hide_index=True, use_container_width=True)

    st.subheader(t("survey_title", lang))
    if state.get("survey_done"):
        st.success(t("survey_thanks", lang))
    else:
        with st.form("survey_form"):
            ease = st.slider(t("survey_ease", lang), 1, 5, 3)
            useful = st.slider(t("survey_useful", lang), 1, 5, 3)
            design = st.slider(t("survey_design", lang), 1, 5, 3)
            comment = st.text_area(t("survey_comment", lang))
            if st.form_submit_button(t("survey_send", lang), use_container_width=True):
                db.log_survey(config.DB_PATH, ease, useful, design, comment)
                state.survey_done = True
                st.rerun()
