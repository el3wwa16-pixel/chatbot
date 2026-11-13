"""The website. Run with:  streamlit run app.py"""
import json
import os
import re
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import brand
import config
from bot import db, extras, planner, tools
from bot.agent import TourismAgent
from bot.cities import CITIES, canonical_city
from bot.cloud_llm import DEFAULT_BASE_URL, DEFAULT_MODEL, CloudClient
from bot.faq import FAQ
from bot.i18n import t
from bot.llm import OllamaClient, OllamaError
from bot.rag import KnowledgeBase
from bot.simple import LOCAL_MATCH, SimpleAgent, format_booking, is_english

st.set_page_config(page_title=f"{config.BOT_NAME} | مساعد السياحة في مصر",
                   page_icon=brand.ICON, layout="centered")
st.markdown(brand.css(), unsafe_allow_html=True)

EXAMPLES = {
    "ar": ["أروح فين في إسكندرية في يومين؟", "احكيلي عن قلعة قايتباي", "الجو عامل إيه في أسوان؟",
           "عايز فندق في الأقصر بأقل من 2000 جنيه", "عندي شكوى وعايز أكلم موظف"],
    "en": ["Tell me about Karnak temple", "What is the weather in Aswan?",
           "What tours can I book in Luxor?", "Hotels in Alexandria under 3000"],
}
PAGES = ["chat", "plan", "budget", "money", "phrases", "stats"]
WEB_TOOL = "search_web_knowledge"


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
    files = list((config.BASE_DIR / "bot").glob("*.py")) + [config.KNOWLEDGE_PATH, config.FAQ_PATH]
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


bot = load_bot(code_version(), secret("CLOUD_API_KEY") or secret("GROQ_API_KEY"),
               secret("CLOUD_MODEL") or DEFAULT_MODEL, secret("CLOUD_BASE_URL") or DEFAULT_BASE_URL)
kb, faq, cloud = bot["kb"], bot["faq"], bot["cloud"]
local_ready = bot["local"].is_up() and bot["local"].has_model(config.CHAT_MODEL)
mode = "cloud" if cloud else "local" if local_ready else "simple"

state = st.session_state
state.setdefault("lang", "ar")
state.setdefault("nav", "chat")
state.setdefault("messages", [])
lang = state.lang


def ask(question: str) -> None:
    """Send a question to the chat (from a button on any page)."""
    state.pending = question
    state.nav = "chat"


# ---- answering -------------------------------------------------------------
def media_for(question: str) -> dict:
    """A photo and map points when the question is clearly about a known place."""
    hits = kb.search(question, k=1, prefer_city=canonical_city(question))
    if not hits or kb.match_strength(question, hits[0]["doc"]) < LOCAL_MATCH:
        return {}
    doc = hits[0]["doc"]
    by_id = {d["id"]: d for d in kb.docs}
    spots = [by_id[i] for i in doc.get("stops", []) if i in by_id] or [doc]
    points = [{"name": d["title_ar"], "lat": d["lat"], "lon": d["lon"]} for d in spots if "lat" in d]
    image = cached_image(doc["wiki"]) if doc.get("wiki") and not doc.get("stops") else None
    return {"points": points, "image": image}


def answer(history: list[dict], question: str) -> dict:
    stored = faq.match(question)  # a frequently asked question: answer it as written
    if stored:
        db.log_question(config.DB_PATH, question, "faq")
        return {"role": "assistant", "content": stored, "question": question}
    used_model = mode != "simple"
    try:
        if mode == "cloud":
            result = bot["cloud_agent"].reply(history, question)
        elif mode == "local":
            result = bot["local_agent"].reply(history, question)
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
    with st.expander(t("why_name", lang)):
        st.markdown(t("why_name_text", lang))
    with st.expander(t("sources", lang)):
        st.markdown(f"- {t('src_knowledge', lang)} ({len(kb.docs)})\n- {t('src_faq', lang)} ({len(faq.entries)})\n"
                    "- Wikipedia (ar / en), Wikivoyage\n- Open-Meteo, wttr.in\n- open.er-api.com\n- SQLite")

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
        st.rerun()

# ---- header and navigation ---------------------------------------------------
st.markdown(brand.hero_html(config.BOT_NAME, t("tagline", lang), latin="DEMO AI"),
            unsafe_allow_html=True)
page = st.radio("nav", PAGES, key="nav", horizontal=True, label_visibility="collapsed",
                format_func=lambda p: t(f"nav_{p}", lang))


def show_map(points: list[dict]) -> None:
    if points:
        st.map(pd.DataFrame(points), latitude="lat", longitude="lon", height=260,
               use_container_width=True)


# ---- chat ------------------------------------------------------------------
if page == "chat":
    with st.chat_message("assistant", avatar=brand.ICON):
        st.markdown(t("welcome", lang, name=config.BOT_NAME))

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
        if message.get("question"):
            left, right = st.columns([1, 3])
            with left:
                if hasattr(st, "feedback"):
                    rating = st.feedback("thumbs", key=f"fb_{i}")
                    if rating is not None and not message.get("rated"):
                        db.log_feedback(config.DB_PATH, message["question"], rating)
                        message["rated"] = True
            with right:
                if i == len(state.messages) - 1:
                    listen_button(message["content"], lang)

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

# ---- trip planner ------------------------------------------------------------
elif page == "plan":
    city = st.selectbox(t("city", lang), list(CITIES),
                        format_func=lambda c: CITIES[c]["ar"] if lang == "ar" else c)
    days = st.slider(t("days", lang), 1, 7, 2)
    interests = st.multiselect(t("interests", lang), list(planner.INTERESTS), default=["history"],
                               format_func=lambda k: planner.INTERESTS[k][lang])
    if st.button(t("make_plan", lang), use_container_width=True):
        state.plan = planner.build_plan(kb.docs, config.DB_PATH, city, days, interests)
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

# ---- budget ------------------------------------------------------------------
elif page == "budget":
    st.caption(t("budget_intro", lang))
    city = st.selectbox(t("city", lang), list(CITIES),
                        format_func=lambda c: CITIES[c]["ar"] if lang == "ar" else c)
    hotels = db.query(config.DB_PATH,
                      "SELECT * FROM hotels WHERE city = ? ORDER BY price_per_night_egp", (city,))
    city_tours = db.query(config.DB_PATH,
                          "SELECT * FROM tours WHERE city = ? ORDER BY price_per_person_egp", (city,))
    hotel = st.selectbox(t("hotel", lang), [None] + hotels,
                         format_func=lambda h: t("no_hotel", lang) if h is None
                         else f"{h['name']} · {h['price_per_night_egp']:,} {t('egp', lang)}")
    left, right = st.columns(2)
    nights = left.number_input(t("nights", lang), 1, 30, 3)
    people = right.number_input(t("people", lang), 1, 20, 2)
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

# ---- currency ----------------------------------------------------------------
elif page == "money":
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

# ---- phrasebook --------------------------------------------------------------
elif page == "phrases":
    st.markdown(f"| {t('phrase_ar', lang)} | {t('phrase_say', lang)} | {t('phrase_en', lang)} |\n"
                "|---|---|---|\n" + "\n".join(f"| {ar} | {say} | {en} |" for ar, say, en in extras.PHRASES))

# ---- statistics --------------------------------------------------------------
else:
    numbers = db.stats(config.DB_PATH)
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

    # Question texts come from other visitors, so they need the admin password.
    admin = secret("ADMIN_PASSWORD")
    if admin and st.text_input(t("st_password", lang), type="password") == admin:
        for title, rows in ((t("st_top", lang), numbers["top_questions"]),
                            (t("st_missing", lang), numbers["unanswered_list"]),
                            (t("st_disliked", lang), numbers["disliked_list"])):
            st.subheader(title)
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.caption(t("st_empty", lang))
    else:
        st.caption(t("st_locked", lang))
