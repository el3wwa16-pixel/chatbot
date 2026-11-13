"""Simple mode: the bot without a language model.

Used automatically when Ollama or the chat model is not available, so the project
can always be demonstrated. It detects the intent with keywords, then uses the
same knowledge base and the same tools as the AI mode. It cannot hold a free
conversation or write custom itineraries; that is what the model adds.
"""
import re

from .agent import AgentResult
from .cities import CITIES, canonical_city
from .textutil import normalize
from .tools import dispatch

_WORD = re.compile(r"[^\W_]+")
LOCAL_MATCH = 0.45  # below this, the local topic is a weak match: ask the web sources
_REF = re.compile(r"TB-[A-Z0-9]{6}")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

GREET = {"اهلا", "مرحبا", "السلام", "سلام", "hi", "hello", "hey", "ازيك", "صباح", "مساء", "هاي"}
THANKS = {"شكرا", "متشكر", "تسلم", "thanks", "thank", "ميرسي"}
WEATHER = {"الجو", "جو", "طقس", "الطقس", "حراره", "الحراره", "مطر", "weather",
           "temperature", "forecast", "rain", "hot", "cold"}
CANCEL = ("الغ", "كنسل", "cancel")
HANDOFF = ("شكوي", "شكاوي", "موظف", "استرداد", "refund", "complain", "human", "representative")
HOTEL = ("فندق", "فنادق", "اوتيل", "منتجع", "hotel", "resort", "accommodation", "اقامه")
TOUR = ("جوله", "جولات", "رحله", "رحلات", "tour", "excursion", "activities")
PLAN = ("برنامج", "itinerary", "plan", "خطه")
BOOK = ("احجز", "حجز", "book", "reserve")
CHEAPER = ("اقل", "تحت", "ارخص", "under", "below", "less", "max", "budget", "اقصي")


def is_english(text: str) -> bool:
    text = _REF.sub("", text.upper())  # a booking code is not English
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return latin > arabic


def format_booking(b: dict, en: bool = False) -> str:
    """Readable confirmation for a create_booking result or a bookings row."""
    item = b.get("item") or b.get("item_name", "")
    nights = b.get("nights") or 0
    if en:
        lines = [f"**Booking {b['ref']}** ({b.get('status', '')})", f"- {item}",
                 f"- Name: {b['customer_name']}", f"- Date: {b['start_date']}",
                 f"- People: {b['people']}" + (f", nights: {nights}" if nights else ""),
                 f"- Total: {b['total_egp']:,} EGP",
                 "", "Demo booking: no real reservation or payment."]
    else:
        lines = [f"**الحجز {b['ref']}** ({b.get('status', '')})", f"- {item}",
                 f"- الاسم: {b['customer_name']}", f"- التاريخ: {b['start_date']}",
                 f"- عدد الأفراد: {b['people']}" + (f"، الليالي: {nights}" if nights else ""),
                 f"- الإجمالي: {b['total_egp']:,} جنيه",
                 "", "حجز تجريبي: لا يوجد حجز فعلي أو دفع حقيقي."]
    return "\n".join(lines)


class SimpleAgent:
    def __init__(self, kb, db_path):
        self.kb = kb
        self.db_path = db_path

    # ---- helpers ------------------------------------------------------
    def _tool(self, result: AgentResult, name: str, **args) -> dict:
        out = dispatch(name, args, self.db_path)
        result.tool_events.append({"tool": name, "args": args, "result": out})
        return out

    @staticmethod
    def _city(text: str, history: list[dict]) -> str | None:
        city = canonical_city(text)
        if city:
            return city
        for m in reversed(history):
            if m.get("role") == "user":
                city = canonical_city(m.get("content", ""))
                if city:
                    return city
        return None

    @staticmethod
    def _max_price(n: str) -> int | None:
        if not any(k in n for k in CHEAPER):
            return None
        numbers = [int(x) for x in re.findall(r"\d{3,6}", n.translate(_DIGITS))]
        return max(numbers) if numbers else None

    # ---- main ---------------------------------------------------------
    def reply(self, history: list[dict], user_text: str) -> AgentResult:
        result = AgentResult(text="", search_mode=self.kb.mode)
        en = is_english(user_text)
        n = normalize(user_text)
        words = set(_WORD.findall(n))
        has = lambda keys: any(k in n for k in keys)  # noqa: E731
        city = self._city(user_text, history)
        city_ar = CITIES[city]["ar"] if city else ""

        ref = _REF.search(user_text.upper())
        if ref:
            if has(CANCEL):
                out = self._tool(result, "cancel_booking", ref=ref.group())
                if "error" in out:
                    result.text = (f"Could not cancel: {out['error']}." if en else
                                   f"مقدرتش ألغي الحجز {ref.group()}: الرقم غير موجود أو ملغي بالفعل.")
                else:
                    result.text = (f"Booking {out['ref']} ({out['item']}) is cancelled." if en else
                                   f"تم إلغاء الحجز {out['ref']} ({out['item']}).")
            else:
                out = self._tool(result, "get_booking", ref=ref.group())
                result.text = (("No booking found with that reference." if en else
                                "مفيش حجز بالرقم ده. راجع الرقم المرجعي.")
                               if "error" in out else format_booking(out, en))
            return result

        if has(HANDOFF):
            out = self._tool(result, "handoff_to_human", reason=user_text.strip()[:200])
            result.text = (f"I opened ticket #{out['ticket']} for our customer service team. "
                           "A human agent will follow up with you." if en else
                           f"فتحتلك تذكرة رقم #{out['ticket']} لفريق خدمة العملاء، "
                           "وموظف هيتواصل معاك. للطوارئ: شرطة السياحة 126.")
            return result

        if words & WEATHER:
            if not city:
                result.text = "Which city?" if en else "في أي مدينة؟ مثلًا: الجو في أسوان."
                return result
            out = self._tool(result, "get_weather", city=city)
            if "error" in out:
                result.text = ("The weather service needs an internet connection." if en else
                               "خدمة الطقس محتاجة إنترنت ومش متاحة دلوقتي.")
                return result
            now = out["now"]
            head = (f"**Weather in {city} now:** {now['temperature_c']}°C, wind {now['wind_kmh']} km/h"
                    if en else
                    f"**الجو في {city_ar} دلوقتي:** {now['temperature_c']}° مئوية، "
                    f"{now['condition']}، رياح {now['wind_kmh']} كم/س")
            days = [(f"- {d['date']}: {d['min_c']}° to {d['max_c']}°, rain {d['rain_chance_percent']}%"
                     if en else
                     f"- {d['date']}: من {d['min_c']}° إلى {d['max_c']}°، احتمال المطر {d['rain_chance_percent']}%")
                    for d in out["forecast"]]
            result.text = "\n".join([head, ""] + days)
            return result

        hint = ("\n\nTo book, use the **quick booking** form in the sidebar." if en else
                "\n\nللحجز استخدم نموذج **حجز سريع** في الشريط الجانبي.")

        if has(HOTEL):
            if not city:
                result.text = ("Which city? " if en else "في أي مدينة؟ المدن المتاحة: ") + \
                    ("، ".join(c["ar"] for c in CITIES.values()) if not en else ", ".join(CITIES))
                return result
            args = {"city": city}
            price = self._max_price(n)
            if price:
                args["max_price_egp"] = price
            out = self._tool(result, "search_hotels", **args)
            if not out["hotels"]:
                result.text = ("No hotels match that budget there." if en else
                               f"مفيش فنادق في {city_ar} بالميزانية دي. جرّب ميزانية أعلى.")
                return result
            head = f"**Hotels in {city}:**" if en else f"**فنادق في {city_ar}:**"
            unit = "EGP/night" if en else "جنيه/الليلة"
            rows = [f"- **{h['name']}** · {h['stars']}★ · {h['price_per_night_egp']:,} {unit}\n"
                    f"  {h['description']}" for h in out["hotels"]]
            result.text = "\n".join([head, ""] + rows) + hint
            return result

        if has(TOUR) and not has(PLAN):
            out = self._tool(result, "search_tours", city=city or "")
            head = (f"**Tours{' in ' + city if city else ''}:**" if en else
                    f"**جولات{' في ' + city_ar if city else ' متاحة'}:**")
            unit = "EGP/person" if en else "جنيه/الفرد"
            hours = "h" if en else "ساعة"
            rows = [f"- **{t['name']}** ({t['city']}) · {t['duration_hours']:g} {hours} · "
                    f"{t['price_per_person_egp']:,} {unit}\n  {t['description']}"
                    for t in out["tours"]]
            result.text = "\n".join([head, ""] + rows) + hint
            return result

        if has(BOOK):
            result.text = ("Tell me what you need, for example: hotels in Luxor, or tours in Aswan."
                           if en else
                           "قولي عايز تحجز إيه، مثلًا: فندق في الأقصر، أو جولات في أسوان.") + hint
            return result

        if len(words) <= 4 and words & GREET:
            result.text = ("Hello! Ask me about places in Egypt, the weather, hotels or tours."
                           if en else
                           "أهلًا بيك! اسألني عن أي مكان في مصر، أو الجو، أو الفنادق والجولات.")
            return result
        if len(words) <= 4 and words & THANKS:
            result.text = "You're welcome!" if en else "العفو! تحت أمرك في أي وقت."
            return result

        hits = self.kb.search(user_text, k=3, prefer_city=canonical_city(user_text))
        strong = bool(hits) and self.kb.match_strength(user_text, hits[0]["doc"]) >= LOCAL_MATCH
        if not strong:
            web = self._tool(result, "search_web_knowledge", query=user_text,
                             lang="en" if en else "ar")
            if "error" not in web:
                result.sources = [f"{web['source']}: {web['title']}"]
                label = "Source" if en else "المصدر"
                result.text = (f"**{web['title']}**\n\n{web['extract']}\n\n"
                               f"{label}: [{web['source']}]({web['url']})")
                return result
        result.sources = [h["doc"].get("title_ar", "") for h in hits]
        if not hits:
            result.answered = False
            result.text = (
                "I could not find that in my knowledge base. Try a place name such as Karnak, "
                "or ask about the weather, hotels or tours." if en else
                "ملقتش إجابة في قاعدة المعرفة. جرّب اسم مكان زي الكرنك أو قلعة قايتباي، "
                "أو اسأل عن الجو أو الفنادق أو الجولات.")
            return result
        top = hits[0]["doc"]
        # Straight to the answer; English readers also get the English name of the topic.
        result.text = f"**{top['title_en']}**\n\n{top['text']}" if en else top["text"]
        return result
