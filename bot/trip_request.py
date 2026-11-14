"""Understands a whole trip request in one sentence and answers it with a plan.

Example: "3 أيام في الأقصر بميزانية 5000 جنيه مع طفلين" -> city, days, budget and
children are read from the text, then a day-by-day plan and a cost estimate are built.
"""
import math
import re

from . import db, planner
from .cities import CITIES, canonical_city
from .textutil import normalize

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DAY_WORDS = [("اسبوعين", 14), ("اسبوع", 7), ("week", 7), ("يومين", 2), ("ليلتين", 2),
              ("يوم واحد", 1), ("one day", 1), ("weekend", 2)]
_NUMBER_WORDS = {"تلات": 3, "ثلاث": 3, "تلاته": 3, "اربع": 4, "اربعه": 4, "خمس": 5, "خمسه": 5,
                 "ست": 6, "سبع": 7, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "two": 2}
_DAYS = re.compile(r"(\d{1,2})\s*(?:ايام|يوم|ليالي|ليله|days?|nights?)")
_PEOPLE = re.compile(r"(\d{1,2})\s*(?:افراد|اشخاص|انفار|people|persons|adults|travell?ers)")
_KIDS_N = re.compile(r"(\d{1,2})\s*(?:اطفال|kids|children)")
_HOT = ("الحر", "حر", "heat", "hot")
_SOUTH = {"Luxor", "Aswan"}


def parse(text: str) -> dict | None:
    """Returns {'city','days','budget','people','kids','avoid_heat'} for a full trip request,
    or None when the message is not one (needs a city, a length, and a budget or party size)."""
    n = normalize(text).translate(_DIGITS)
    city = canonical_city(text)
    if not city:
        return None

    days = None
    match = _DAYS.search(n)
    if match:
        days = int(match.group(1))
    else:
        for word, value in _DAY_WORDS:
            if word in n:
                days = value
                break
        else:
            for word, value in _NUMBER_WORDS.items():
                if re.search(rf"\b{word}\s*(?:ايام|ليالي|days|nights)", n):
                    days = value
                    break
    if not days:
        return None

    kids = 0
    match = _KIDS_N.search(n)
    if match:
        kids = int(match.group(1))
    elif "طفلين" in n or "two kids" in n or "two children" in n:
        kids = 2
    elif re.search(r"\bطفل\b|\bchild\b|\bkid\b", n):
        kids = 1
    elif any(w in n for w in ("اطفال", "اولادي", "عيالي", "kids", "children")):
        kids = 1

    match = _PEOPLE.search(n)
    people = int(match.group(1)) if match else None

    used = {str(days), str(kids), str(people)}
    amounts = [int(x) for x in re.findall(r"\d{3,7}", n) if x not in used and int(x) >= 500]
    budget = max(amounts) if amounts else None

    if budget is None and not kids and people is None:
        return None
    return {"city": city, "days": max(1, min(days, 7)), "budget": budget,
            "people": people or (2 if kids else 1), "kids": kids,
            "avoid_heat": any(w in n.split() for w in _HOT)}


def compose(kb_docs: list[dict], db_path, request: dict, en: bool = False) -> str:
    city, days, kids = request["city"], request["days"], request["kids"]
    travellers = request["people"] + kids
    interests = ["nature", "culture", "history"] if kids else ["history"]
    plan = planner.build_plan(kb_docs, db_path, city, days, interests)
    city_label = city if en else CITIES[city]["ar"]

    lines = [f"**{days} days in {city_label} for {travellers} people**" if en
             else f"**{days} أيام في {city_label} لـ {travellers} أفراد**", ""]
    for number, day in enumerate(plan["days"], 1):
        names = "، ".join(i["title_en"] if en else i["title"] for i in day)
        free = "free day" if en else "يوم حر"
        lines.append(f"- {'Day' if en else 'اليوم'} {number}: {names or free}")

    nights = max(1, days - 1)
    rooms = math.ceil(travellers / 2)
    hotels = db.query(db_path, "SELECT * FROM hotels WHERE city = ? ORDER BY price_per_night_egp", (city,))
    if hotels:
        budget = request["budget"]
        affordable = [h for h in hotels if budget is None or h["price_per_night_egp"] * nights * rooms <= 0.6 * budget]
        hotel = affordable[-1] if affordable else hotels[0]  # the best one that leaves room for the rest
        cost = hotel["price_per_night_egp"] * nights * rooms
        lines += ["", (f"- **Stay:** {hotel['name']}, {nights} nights × {rooms} rooms = {cost:,} EGP" if en else
                       f"- **الإقامة:** {hotel['name']}، {nights} ليالٍ × {rooms} غرف = {cost:,} جنيه")]
        if budget is not None:
            left = budget - cost
            if left >= 0:
                lines.append(f"- **Left for tours, food and transport:** {left:,} EGP" if en else
                             f"- **الباقي للجولات والأكل والمواصلات:** {left:,} جنيه")
            else:
                lines.append(f"- The cheapest stay is {-left:,} EGP over your budget. Try fewer nights or a higher budget."
                             if en else f"- أرخص إقامة أعلى من ميزانيتك بـ {-left:,} جنيه. جرّب ليالي أقل أو ميزانية أعلى.")
    if kids:
        lines += ["", "With children: start early, rest at midday, and pick a hotel with a pool." if en
                  else "مع الأطفال: ابدأ بدري، ارتاح وقت الظهر، واختار فندق فيه حمام سباحة."]
    if request["avoid_heat"] and city in _SOUTH:
        lines += ["", f"{city_label} is very hot from May to September. Visit between October and April, "
                      "and see the sites before 10 am." if en else
                  f"{city_label} حر جدًا من مايو لسبتمبر. الأفضل تزورها من أكتوبر لأبريل، وتخلص المزارات قبل 10 الصبح."]
    lines += ["", "Prices are demo figures from this site's catalogue." if en
              else "الأسعار تجريبية من كتالوج الموقع."]
    return "\n".join(lines)
