"""Trip planner: a day-by-day plan built from the knowledge base and the tours list."""
from . import db
from .cities import CITIES

INTERESTS = {
    "history": {"ar": "آثار ومتاحف", "en": "History and museums",
                "categories": {"آثار فرعونية", "آثار إسلامية", "آثار قبطية", "آثار رومانية",
                               "متاحف", "معالم", "تاريخ حديث", "دينية وطبيعة"}},
    "nature": {"ar": "بحر وطبيعة", "en": "Sea and nature",
               "categories": {"شواطئ وغوص", "طبيعة", "واحات وصحراء"}},
    "culture": {"ar": "ثقافة وترفيه", "en": "Culture and leisure",
                "categories": {"ثقافة", "ترفيه", "رحلات"}},
    "food": {"ar": "أكل وأسواق", "en": "Food and markets", "categories": {"أكل", "أسواق"}},
}
SLOTS = {"ar": ["الصبح", "بعد الظهر", "المغرب"], "en": ["Morning", "Afternoon", "Evening"]}
# Places that belong to a city's day trips even though they are filed under a neighbour.
NEARBY = {"Cairo": ["Giza"], "Giza": ["Cairo"], "Dahab": ["South Sinai"], "Sharm El Sheikh": ["South Sinai"]}
_SKIP = {"برامج", "معلومات عملية", "خدمة العملاء"}


def build_plan(kb_docs: list[dict], db_path, city: str, days: int, interests: list[str]) -> dict:
    """Returns {'city', 'days': [[{'slot','title','note','kind','lat'?,'lon'?}]], 'points': [...]}."""
    days = max(1, min(int(days), 7))
    wanted = set().union(*(INTERESTS[i]["categories"] for i in interests if i in INTERESTS))
    area = [city] + NEARBY.get(city, [])
    places = [d for d in kb_docs if d.get("city") in area and d.get("category") not in _SKIP]
    places.sort(key=lambda d: (d.get("city") != city,))  # the city itself first, neighbours after
    chosen = [d for d in places if d.get("category") in wanted] if wanted else []
    rest = [d for d in places if d not in chosen]

    items = [{"title": d["title_ar"], "title_en": d["title_en"], "kind": "place",
              "note": d["text"].split(".")[0].strip() + ".", "lat": d.get("lat"), "lon": d.get("lon")}
             for d in chosen + rest]
    tours = db.query(db_path, "SELECT * FROM tours WHERE city = ? ORDER BY price_per_person_egp", (city,))
    items += [{"title": t["name"], "title_en": t["name"], "kind": "tour",
               "note": f"{t['description']} ({t['price_per_person_egp']:,} جنيه للفرد)",
               "lat": None, "lon": None} for t in tours]

    per_day = 3
    plan, used = [], 0
    for _ in range(days):
        day = items[used:used + per_day]
        used += len(day)
        plan.append([{**item, "slot": slot} for slot, item in enumerate(day)])
    points = [{"name": i["title"], "lat": i["lat"], "lon": i["lon"]}
              for day in plan for i in day if i["lat"] is not None]
    return {"city": city, "city_ar": CITIES.get(city, {}).get("ar", city), "days": plan,
            "points": points, "enough": used >= days}  # False: fewer ideas than days
