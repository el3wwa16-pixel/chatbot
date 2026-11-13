"""Cities the bot knows, with Arabic/English aliases and coordinates."""
from .textutil import normalize

CITIES = {
    "Cairo": {"ar": "القاهرة", "lat": 30.0444, "lon": 31.2357,
              "aliases": ["cairo", "القاهرة", "قاهرة", "كايرو"]},
    "Giza": {"ar": "الجيزة", "lat": 30.0131, "lon": 31.2089,
             "aliases": ["giza", "الجيزة", "جيزة", "الاهرامات", "الهرم"]},
    "Alexandria": {"ar": "الإسكندرية", "lat": 31.2001, "lon": 29.9187,
                   "aliases": ["alexandria", "alex", "الإسكندرية", "اسكندرية", "سكندرية"]},
    "Luxor": {"ar": "الأقصر", "lat": 25.6872, "lon": 32.6396,
              "aliases": ["luxor", "الأقصر", "اقصر"]},
    "Aswan": {"ar": "أسوان", "lat": 24.0889, "lon": 32.8998,
              "aliases": ["aswan", "أسوان", "اسوان"]},
    "Hurghada": {"ar": "الغردقة", "lat": 27.2579, "lon": 33.8116,
                 "aliases": ["hurghada", "الغردقة", "غردقة"]},
    "Sharm El Sheikh": {"ar": "شرم الشيخ", "lat": 27.9158, "lon": 34.3300,
                        "aliases": ["sharm el sheikh", "sharm", "شرم الشيخ", "شرم"]},
    "Dahab": {"ar": "دهب", "lat": 28.5091, "lon": 34.5136,
              "aliases": ["dahab", "دهب"]},
    "Siwa": {"ar": "سيوة", "lat": 29.2032, "lon": 25.5195,
             "aliases": ["siwa", "سيوة", "واحة سيوة"]},
    "Marsa Alam": {"ar": "مرسى علم", "lat": 25.0676, "lon": 34.8790,
                   "aliases": ["marsa alam", "مرسى علم", "مرسي علم"]},
    "Fayoum": {"ar": "الفيوم", "lat": 29.3084, "lon": 30.8428,
               "aliases": ["fayoum", "faiyum", "fayyum", "الفيوم", "فيوم"]},
}

_ALIASES = [(normalize(a), name) for name, c in CITIES.items() for a in c["aliases"]]
_ALIASES.sort(key=lambda p: -len(p[0]))  # longest alias first


def canonical_city(name) -> str | None:
    """Map 'اسكندرية' / 'Alex' / 'alexandria' -> 'Alexandria'. None if unknown."""
    if not name:
        return None
    n = normalize(name).strip()
    if len(n) < 3:
        return None
    for alias, city in _ALIASES:
        if alias in n or n in alias:
            return city
    return None
