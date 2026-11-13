"""Chat interface. Run with:  streamlit run app.py"""
from datetime import date

import streamlit as st

import config
from bot import db, tools
from bot.agent import TourismAgent
from bot.llm import OllamaClient, OllamaError
from bot.rag import KnowledgeBase
from bot.simple import SimpleAgent, format_booking

st.set_page_config(page_title=f"{config.BOT_NAME} | مساعد السياحة في مصر",
                   page_icon="🧭", layout="centered")

# Each paragraph picks its own direction: Arabic right-to-left, English left-to-right.
st.markdown("""
<style>
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stSidebar"] p, h1, h2, h3, [data-testid="stCaptionContainer"] {
    unicode-bidi: plaintext; text-align: start;
}
[data-testid="stChatInput"] textarea { unicode-bidi: plaintext; }
</style>
""", unsafe_allow_html=True)

WELCOME = (f"أهلًا بيك! أنا **{config.BOT_NAME}**، مساعدك السياحي في مصر. 🧭\n\n"
           "اسألني عن أي مكان، اطلب برنامج رحلة، اعرف الجو، أو احجز فندق أو جولة.\n\n"
           "You can also write to me in English.")

EXAMPLES = [
    "أروح فين في إسكندرية في يومين؟",
    "احكيلي عن قلعة قايتباي",
    "الجو عامل إيه في أسوان؟",
    "عايز فندق في الأقصر بأقل من 2000 جنيه",
    "What tours can I book in Luxor?",
    "عندي شكوى وعايز أكلم موظف",
]


@st.cache_resource
def load_bot():
    client = OllamaClient(config.OLLAMA_HOST, config.CHAT_MODEL, config.EMBED_MODEL,
                          num_ctx=config.NUM_CTX, temperature=config.TEMPERATURE)
    db.init_db(config.DB_PATH)
    kb = KnowledgeBase(config.KNOWLEDGE_PATH, client=client, cache_path=config.EMB_CACHE_PATH)
    return client, kb, TourismAgent(client, kb, config.DB_PATH), SimpleAgent(kb, config.DB_PATH)


client, kb, ai_agent, simple_agent = load_bot()
ai_ready = client.is_up() and client.has_model(config.CHAT_MODEL)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME}]

# ---- sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("حالة النظام")
    if ai_ready:
        st.success(f"وضع الذكاء الاصطناعي: {config.CHAT_MODEL}")
        if client.has_model(config.EMBED_MODEL):
            st.caption(f"البحث الدلالي: {config.EMBED_MODEL}")
        else:
            st.caption(f"البحث بالكلمات (لبحث أدق: ollama pull {config.EMBED_MODEL})")
    else:
        st.warning("الوضع البسيط: البوت شغال بدون موديل (بحث + أدوات فقط).")
        st.caption("لتشغيل الذكاء الاصطناعي: افتح Ollama ثم نفّذ")
        st.code(f"ollama pull {config.CHAT_MODEL}")
    st.caption(f"قاعدة المعرفة: {len(kb.docs)} موضوع")
    with st.expander("مصادر البيانات"):
        st.markdown(
            f"- قاعدة المعرفة المحلية ({len(kb.docs)} موضوعًا)\n"
            "- ويكيبيديا العربية\n- Wikipedia الإنجليزية\n- Wikivoyage (دليل سفر)\n"
            "- Open-Meteo و wttr.in (الطقس)\n- قاعدة بيانات الحجوزات (SQLite)")

    st.header("جرّب")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example

    st.header("حجز سريع")
    kind_label = st.radio("النوع", ["فندق", "جولة"], horizontal=True)
    kind = "hotel" if kind_label == "فندق" else "tour"
    catalogue = db.query(config.DB_PATH,
                         f"SELECT * FROM {'hotels' if kind == 'hotel' else 'tours'} "
                         "ORDER BY city, name")
    with st.form("booking_form", clear_on_submit=True):
        item = st.selectbox("الاختيار", catalogue,
                            format_func=lambda r: f"{r['city']} · {r['name']}")
        customer = st.text_input("اسم العميل")
        day = st.date_input("التاريخ", value=date.today(), min_value=date.today())
        nights = st.number_input("عدد الليالي", 1, 30, 1) if kind == "hotel" else 1
        people = st.number_input("عدد الأفراد", 1, 20, 2)
        if st.form_submit_button("احجز", use_container_width=True):
            if not customer.strip():
                st.error("اكتب اسم العميل.")
            else:
                booking = tools.create_booking(config.DB_PATH, kind, item["id"], customer,
                                               day.isoformat(), nights=nights, people=people)
                if "error" in booking:
                    st.error(booking["error"])
                else:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": format_booking(booking)})
                    st.rerun()

    st.header("الحجوزات")
    bookings = db.list_bookings(config.DB_PATH)
    if bookings:
        st.dataframe(
            [{"الرقم": b["ref"], "الحجز": b["item_name"], "العميل": b["customer_name"],
              "التاريخ": b["start_date"], "الإجمالي": b["total_egp"], "الحالة": b["status"]}
             for b in bookings], hide_index=True, use_container_width=True)
    else:
        st.caption("لا توجد حجوزات بعد.")

    handoffs = db.list_handoffs(config.DB_PATH)
    if handoffs:
        st.header("تذاكر خدمة العملاء")
        for h in handoffs:
            st.caption(f"#{h['id']} · {h['reason']}")

    if st.button("محادثة جديدة", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": WELCOME}]
        st.rerun()

# ---- chat ----------------------------------------------------------------
st.title(f"🧭 {config.BOT_NAME}")
st.caption("مساعدك السياحي في مصر: أماكن، برامج رحلات، طقس، وحجز")


def show_details(message: dict) -> None:
    if message.get("sources") or message.get("tool_events"):
        with st.expander("كيف وصل البوت للرد؟"):
            if message.get("mode"):
                st.caption(message["mode"])
            if message.get("sources"):
                st.markdown("**من قاعدة المعرفة:** " + "، ".join(message["sources"]))
            for event in message.get("tool_events", []):
                st.markdown(f"**أداة:** `{event['tool']}`")
                st.json({"args": event["args"], "result": event["result"]}, expanded=False)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_details(message)

prompt = st.chat_input("اكتب سؤالك هنا...") or st.session_state.pop("pending", None)

if prompt:
    history = list(st.session_state.messages[1:])  # skip the welcome message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("بفكر..."):
            mode = f"وضع الذكاء الاصطناعي ({config.CHAT_MODEL})"
            try:
                if not ai_ready:
                    raise OllamaError("model not available")
                result = ai_agent.reply(history, prompt)
            except OllamaError:
                mode = "الوضع البسيط (بدون موديل)"
                result = simple_agent.reply(history, prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": result.sources,
         "tool_events": result.tool_events, "mode": mode})
    st.rerun()
