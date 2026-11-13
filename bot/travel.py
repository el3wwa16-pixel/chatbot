"""Practical travel helpers: distances, routes between cities, packing, holidays, prayer times."""
import math
import re
from datetime import date

import requests

from .web_sources import HEADERS

# ---- distances ---------------------------------------------------------------
def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line (great-circle) distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def nearest(kb_docs: list[dict], lat: float, lon: float, limit: int = 5,
            exclude_id: str | None = None) -> list[dict]:
    """The closest known places, nearest first: [{'doc': ..., 'km': float}]."""
    out = [{"doc": d, "km": round(distance_km(lat, lon, d["lat"], d["lon"]), 1)}
           for d in kb_docs if "lat" in d and d["id"] != exclude_id]
    return sorted(out, key=lambda x: x["km"])[:limit]


# ---- routes between cities (approximate, from general knowledge) ---------------
# (city A, city B): [(mode key, typical duration, note)]
_ROUTES = {
    ("Cairo", "Alexandria"): [("train", "2.5 - 3 h", "قطارات كثيرة يوميًا من محطة رمسيس"), ("bus", "3 h", "نحو 220 كم")],
    ("Cairo", "Luxor"): [("flight", "1 h", ""), ("sleeper", "9 - 10 h", "قطار النوم ليلًا")],
    ("Cairo", "Aswan"): [("flight", "1.5 h", ""), ("sleeper", "12 - 13 h", "قطار النوم ليلًا")],
    ("Luxor", "Aswan"): [("train", "3 h", ""), ("car", "3.5 h", "مع توقف في إدفو وكوم أمبو"), ("cruise", "3 - 4 nights", "رحلة نيلية")],
    ("Cairo", "Hurghada"): [("flight", "1 h", ""), ("bus", "6 h", "نحو 450 كم")],
    ("Cairo", "Sharm El Sheikh"): [("flight", "1 h", ""), ("bus", "7 h", "")],
    ("Sharm El Sheikh", "Dahab"): [("car", "1 - 1.5 h", "نحو 90 كم")],
    ("Cairo", "Dahab"): [("bus", "8 - 9 h", ""), ("flight", "1 h + 1.5 h", "طيران إلى شرم ثم سيارة")],
    ("Cairo", "Siwa"): [("bus", "10 - 12 h", "عن طريق مرسى مطروح، نحو 750 كم")],
    ("Alexandria", "Siwa"): [("bus", "8 - 9 h", "عن طريق مرسى مطروح")],
    ("Cairo", "Fayoum"): [("car", "1.5 - 2 h", "نحو 100 كم")],
    ("Cairo", "Giza"): [("car", "30 - 60 min", "حسب الزحام"), ("metro", "30 min", "خط المترو حتى الجيزة ثم تاكسي للأهرامات")],
    ("Hurghada", "Luxor"): [("car", "4 h", "نحو 300 كم"), ("bus", "4 - 5 h", "")],
    ("Hurghada", "Marsa Alam"): [("car", "3.5 h", "نحو 270 كم")],
    ("Cairo", "Marsa Alam"): [("flight", "1.5 h", "")],
    ("Luxor", "Marsa Alam"): [("car", "4 - 5 h", "")],
}
MODES = {
    "train": {"ar": "قطار", "en": "Train"}, "sleeper": {"ar": "قطار النوم", "en": "Sleeper train"},
    "flight": {"ar": "طيران", "en": "Flight"}, "bus": {"ar": "أتوبيس", "en": "Bus"},
    "car": {"ar": "سيارة", "en": "Car"}, "cruise": {"ar": "رحلة نيلية", "en": "Nile cruise"},
    "metro": {"ar": "مترو", "en": "Metro"},
}


def route(a: str, b: str) -> list[dict]:
    """Ways to travel between two cities, or [] when the pair is not recorded."""
    options = _ROUTES.get((a, b)) or _ROUTES.get((b, a)) or []
    return [{"mode": mode, "duration": duration, "note": note} for mode, duration, note in options]


# ---- packing list --------------------------------------------------------------
_BEACH = {"Hurghada", "Sharm El Sheikh", "Dahab", "Marsa Alam"}
_SOUTH = {"Luxor", "Aswan"}
_DESERT = {"Siwa", "Fayoum"}


def season(month: int) -> str:
    return "winter" if month in (11, 12, 1, 2, 3) else "summer" if month in (5, 6, 7, 8, 9) else "mild"


def packing_list(city: str, month: int, lang: str = "ar") -> list[str]:
    ar = lang == "ar"
    s = season(month)
    items = [("جواز السفر وصورة منه", "Passport and a copy"),
             ("كاش بالجنيه وفكة للبقشيش", "Cash in EGP and small change for tips"),
             ("شاحن وباور بانك", "Charger and power bank"),
             ("واقي شمس ونظارة شمس", "Sunscreen and sunglasses"),
             ("حذاء مريح للمشي", "Comfortable walking shoes")]
    if s == "summer":
        items += [("ملابس قطن خفيفة وقبعة", "Light cotton clothes and a hat"),
                  ("زجاجة مياه", "Water bottle")]
    elif s == "winter":
        items += [("جاكيت خفيف للصبح والليل", "A light jacket for mornings and nights")]
        if city == "Alexandria":
            items += [("مظلة أو جاكيت مطر", "Umbrella or rain jacket")]
    else:
        items += [("ملابس خفيفة مع سويتر للّيل", "Light clothes plus a sweater for the evening")]
    if city in _BEACH:
        items += [("مايوه وفوطة بحر", "Swimwear and a beach towel"),
                  ("حذاء بحر للشعاب المرجانية", "Water shoes for the reefs"),
                  ("واقي شمس آمن على الشعاب", "Reef-safe sunscreen")]
    if city in _SOUTH:
        items += [("قبعة بحواف واسعة", "Wide-brimmed hat")]
        if s == "summer":
            items += [("أملاح تعويض السوائل", "Rehydration salts")]
    if city in _DESERT:
        items += [("ملابس ثقيلة للّيل في الصحراء", "Warm clothes for desert nights"),
                  ("كشاف صغير", "A small torch")]
    if city in ("Cairo", "Giza", "Alexandria"):
        items += [("ملابس تغطي الكتفين والركبتين للمساجد والكنائس",
                   "Clothes covering shoulders and knees for mosques and churches")]
    return [a if ar else e for a, e in items]


# ---- official holidays -----------------------------------------------------------
FIXED_HOLIDAYS = [
    ((1, 7), "عيد الميلاد المجيد (القبطي)", "Coptic Christmas"),
    ((1, 25), "عيد الشرطة وثورة 25 يناير", "Police Day / 25 January Revolution"),
    ((4, 25), "عيد تحرير سيناء", "Sinai Liberation Day"),
    ((5, 1), "عيد العمال", "Labour Day"),
    ((6, 30), "ثورة 30 يونيو", "30 June Revolution"),
    ((7, 23), "ثورة 23 يوليو", "23 July Revolution"),
    ((10, 6), "عيد القوات المسلحة", "Armed Forces Day"),
]
MOVING_HOLIDAYS = [
    ("عيد الفطر", "Eid al-Fitr"), ("عيد الأضحى ووقفة عرفات", "Eid al-Adha and Arafat Day"),
    ("رأس السنة الهجرية", "Islamic New Year"), ("المولد النبوي", "Prophet's Birthday"),
    ("شم النسيم", "Sham El-Nessim (spring festival)"),
]


def upcoming_holidays(today: date, lang: str = "ar") -> list[dict]:
    """Fixed-date official holidays, soonest first."""
    out = []
    for (month, day), ar, en in FIXED_HOLIDAYS:
        when = date(today.year, month, day)
        if when < today:
            when = date(today.year + 1, month, day)
        out.append({"date": when.isoformat(), "name": ar if lang == "ar" else en,
                    "days_left": (when - today).days})
    return sorted(out, key=lambda h: h["days_left"])


# ---- prayer times (aladhan.com: free, no key) --------------------------------------
PRAYERS = [("Fajr", "الفجر"), ("Sunrise", "الشروق"), ("Dhuhr", "الظهر"),
           ("Asr", "العصر"), ("Maghrib", "المغرب"), ("Isha", "العشاء")]


def prayer_times(lat: float, lon: float) -> dict:
    """Today's prayer times (Egyptian General Authority of Survey method)."""
    try:
        data = requests.get("https://api.aladhan.com/v1/timings", headers=HEADERS, timeout=10,
                            params={"latitude": lat, "longitude": lon, "method": 5}).json()
        timings = data["data"]["timings"]
        return {"times": {key: re.sub(r"\s*\(.*\)", "", str(timings[key])) for key, _ in PRAYERS}}
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return {"error": "Prayer times are unavailable right now (needs internet)."}
