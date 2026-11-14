"""Trip planner: a day-by-day plan built from the knowledge base and the tours list.

Places are visited in a sensible order (each stop is the nearest one not yet
visited), every visit has a typical duration, and a day is filled until its
hours run out, so the plan shows start times and the distance between stops.
"""
from . import db
from .cities import CITIES
from .travel import distance_km

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
# Places that belong to a city's day trips even though they are filed under a neighbour.
NEARBY = {"Cairo": ["Giza"], "Giza": ["Cairo"], "Dahab": ["South Sinai"], "Sharm El Sheikh": ["South Sinai"]}
_SKIP = {"برامج", "معلومات عملية", "خدمة العملاء"}

# Typical time on site, in hours, by kind of place (estimates for planning only).
VISIT_HOURS = {"آثار فرعونية": 2.5, "متاحف": 2.0, "آثار إسلامية": 1.5, "آثار قبطية": 1.5,
               "آثار رومانية": 1.0, "معالم": 1.5, "تاريخ حديث": 1.5, "دينية وطبيعة": 3.0,
               "شواطئ وغوص": 4.0, "طبيعة": 4.0, "واحات وصحراء": 5.0, "ثقافة": 1.5,
               "ترفيه": 2.0, "رحلات": 3.0, "أكل": 1.5, "أسواق": 1.5}
DAY_START = 9.0     # first visit at 09:00
DAY_HOURS = 9.0     # visits, travel and lunch fit between 09:00 and 18:00
LUNCH_AT = 12.5     # the first stop that would start after 12:30 waits for lunch
LUNCH_HOURS = 1.0
ROAD_FACTOR = 1.3   # roads are longer than the straight line


def travel_minutes(km: float) -> int:
    """Rough driving time: slow inside a city, faster on the open road."""
    if km <= 0:
        return 0
    speed = 25 if km < 15 else 60
    return max(5, round(km * ROAD_FACTOR / speed * 60 / 5) * 5)


def clock(hours: float) -> str:
    total = round(hours * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


def _in_visit_order(places: list[dict]) -> list[dict]:
    """Nearest-neighbour route: start at the first (most relevant) place, then always go
    to the closest place not yet visited. Places without coordinates keep their order at the end."""
    located = [p for p in places if p.get("lat") is not None]
    rest = [p for p in places if p.get("lat") is None]
    if not located:
        return rest
    route, pool = [located[0]], located[1:]
    while pool:
        here = route[-1]
        nxt = min(pool, key=lambda p: distance_km(here["lat"], here["lon"], p["lat"], p["lon"]))
        pool.remove(nxt)
        route.append(nxt)
    return route + rest


def build_plan(kb_docs: list[dict], db_path, city: str, days: int, interests: list[str]) -> dict:
    """Returns {'city', 'city_ar', 'days': [[item, ...]], 'points', 'total_km', 'enough'}.
    Each item: title, title_en, kind, note, lat, lon, start ('09:00'), hours, km_from_previous,
    minutes_from_previous, after_lunch."""
    days = max(1, min(int(days), 7))
    wanted = set().union(*(INTERESTS[i]["categories"] for i in interests if i in INTERESTS))
    area = [city] + NEARBY.get(city, [])
    places = [d for d in kb_docs if d.get("city") in area and d.get("category") not in _SKIP]
    places.sort(key=lambda d: (d.get("city") != city,))  # the city itself first, neighbours after
    chosen = [d for d in places if d.get("category") in wanted] if wanted else []
    ranked = chosen + [d for d in places if d not in chosen]

    # Keep the most relevant places (about four a day), then put them in visiting order.
    shortlist = [{"title": d["title_ar"], "title_en": d["title_en"], "kind": "place",
                  "note": d["text"].split(".")[0].strip() + ".", "lat": d.get("lat"), "lon": d.get("lon"),
                  "hours": VISIT_HOURS.get(d.get("category"), 1.5)} for d in ranked[:days * 4]]
    route = _in_visit_order(shortlist)
    tours = db.query(db_path, "SELECT * FROM tours WHERE city = ? ORDER BY price_per_person_egp", (city,))
    route += [{"title": t["name"], "title_en": t["name"], "kind": "tour",
               "note": f"{t['description']} ({t['price_per_person_egp']:,} جنيه للفرد)",
               "lat": None, "lon": None, "hours": float(t["duration_hours"])} for t in tours]

    plan, position, total_km = [], 0, 0.0
    for _ in range(days):
        day, used, previous, lunch_done = [], 0.0, None, False
        while position < len(route):
            item = route[position]
            km = (round(distance_km(previous["lat"], previous["lon"], item["lat"], item["lon"]) * ROAD_FACTOR, 1)
                  if previous and item["lat"] is not None and previous["lat"] is not None else 0.0)
            minutes = travel_minutes(km / ROAD_FACTOR) if km else 0
            lunch = LUNCH_HOURS if day and not lunch_done and DAY_START + used >= LUNCH_AT else 0.0
            needed = lunch + minutes / 60 + item["hours"]
            if day and used + needed > DAY_HOURS:
                break  # the day is full; this stop opens the next day
            start = DAY_START + used + lunch + minutes / 60
            day.append({**item, "slot": len(day), "start": clock(start), "after_lunch": bool(lunch),
                        "km_from_previous": km, "minutes_from_previous": minutes})
            lunch_done = lunch_done or bool(lunch)
            used += needed
            total_km += km
            previous = item
            position += 1
        plan.append(day)

    points = [{"name": i["title"], "lat": i["lat"], "lon": i["lon"]}
              for day in plan for i in day if i["lat"] is not None]
    return {"city": city, "city_ar": CITIES.get(city, {}).get("ar", city), "days": plan,
            "points": points, "total_km": round(total_km, 1),
            "enough": all(plan)}  # False: fewer ideas than days, so some days are empty


def legs(points: list[dict]) -> list[dict]:
    """Distance and rough driving time between consecutive points of a route."""
    out = []
    for a, b in zip(points, points[1:]):
        km = distance_km(a["lat"], a["lon"], b["lat"], b["lon"])
        out.append({"from": a.get("name", ""), "to": b.get("name", ""),
                    "km": round(km * ROAD_FACTOR, 1), "minutes": travel_minutes(km)})
    return out
