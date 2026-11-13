"""Small tools around the chat: budget, currency, phrasebook, place photos, tickets."""
import html
import math

import requests

from . import db
from .web_sources import HEADERS

# ---- phrasebook ----------------------------------------------------------
PHRASES = [
    ("أهلًا", "ahlan", "Hello"),
    ("صباح الخير", "sabah el-kheir", "Good morning"),
    ("مساء الخير", "masa' el-kheir", "Good evening"),
    ("شكرًا", "shukran", "Thank you"),
    ("من فضلك", "min fadlak (to a man) / min fadlik (to a woman)", "Please"),
    ("أيوه / لأ", "aywa / la'", "Yes / No"),
    ("لو سمحت", "law samaht", "Excuse me"),
    ("آسف", "asef", "Sorry"),
    ("بكام ده؟", "bekam da?", "How much is this?"),
    ("غالي قوي", "ghali awi", "Too expensive"),
    ("آخر كلام كام؟", "akher kalam kam?", "What is your final price?"),
    ("مش عايز، شكرًا", "mesh ayez, shukran", "I don't want it, thank you"),
    ("فين الحمام؟", "fein el-hammam?", "Where is the toilet?"),
    ("فين ...؟", "fein ...?", "Where is ...?"),
    ("عايز أروح ...", "ayez aruh ...", "I want to go to ..."),
    ("على طول", "ala tool", "Straight ahead"),
    ("يمين / شمال", "yemin / shemal", "Right / Left"),
    ("هنا كويس", "hena kwayes", "Here is fine (stop here)"),
    ("الحساب لو سمحت", "el-hesab law samaht", "The bill, please"),
    ("مية معدنية", "mayya ma'daneyya", "Mineral water"),
    ("من غير شطة", "men gheir shatta", "Without chilli"),
    ("أنا نباتي", "ana nabati", "I am vegetarian"),
    ("مش فاهم", "mesh fahem", "I don't understand"),
    ("بتتكلم إنجليزي؟", "betetkallem ingilizi?", "Do you speak English?"),
    ("محتاج مساعدة", "mehtag mosa'da", "I need help"),
    ("يلا بينا", "yalla bina", "Let's go"),
    ("تمام", "tamam", "OK / Fine"),
    ("مع السلامة", "ma'a el-salama", "Goodbye"),
]

# ---- budget ----------------------------------------------------------------
def trip_budget(db_path, hotel_id, nights: int, people: int, tour_ids: list) -> dict:
    """Estimated cost from the demo catalogue: rooms hold two people, tours are per person."""
    nights, people = max(0, int(nights)), max(1, int(people))
    lines, total = [], 0
    if hotel_id is not None and nights:
        rows = db.query(db_path, "SELECT * FROM hotels WHERE id = ?", (int(hotel_id),))
        if rows:
            rooms = math.ceil(people / 2)
            cost = rows[0]["price_per_night_egp"] * nights * rooms
            lines.append({"item": rows[0]["name"], "detail": f"{nights} × {rooms}", "cost": cost, "kind": "hotel"})
            total += cost
    for tour_id in tour_ids or []:
        rows = db.query(db_path, "SELECT * FROM tours WHERE id = ?", (int(tour_id),))
        if rows:
            cost = rows[0]["price_per_person_egp"] * people
            lines.append({"item": rows[0]["name"], "detail": f"× {people}", "cost": cost, "kind": "tour"})
            total += cost
    return {"lines": lines, "total_egp": total, "per_person_egp": round(total / people)}


# ---- currency (open.er-api.com: free, no key) --------------------------------
CURRENCIES = ["USD", "EUR", "GBP", "SAR", "AED", "KWD", "EGP"]


def fetch_rates() -> dict:
    """Rates against the US dollar. Returns {'rates': {...}, 'updated': str} or {'error': ...}."""
    try:
        data = requests.get("https://open.er-api.com/v6/latest/USD", headers=HEADERS, timeout=10).json()
        rates = data["rates"]
        if data.get("result") != "success" or "EGP" not in rates:
            raise KeyError("rates")
        return {"rates": {c: float(rates[c]) for c in CURRENCIES if c in rates},
                "updated": str(data.get("time_last_update_utc", ""))[:16]}
    except (requests.RequestException, KeyError, ValueError, TypeError):
        return {"error": "Exchange rates are unavailable right now (needs internet)."}


def convert(amount: float, source: str, target: str, rates: dict) -> float | None:
    if source not in rates or target not in rates or not rates[source]:
        return None
    return round(float(amount) / rates[source] * rates[target], 2)


# ---- place photos (Wikipedia page image; only freely licensed files) ------------
def place_image(wiki_title: str) -> dict | None:
    """Returns {'url', 'page'} for the lead photo of an English Wikipedia article."""
    if not wiki_title:
        return None
    try:
        data = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "redirects": 1, "titles": wiki_title,
            "prop": "pageimages|info", "piprop": "thumbnail", "pithumbsize": 800,
            "pilicense": "free", "inprop": "url"}, headers=HEADERS, timeout=10).json()
        for page in (data.get("query") or {}).get("pages", {}).values():
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                return {"url": thumb, "page": page.get("fullurl", "https://en.wikipedia.org/")}
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        pass
    return None


# ---- booking ticket --------------------------------------------------------
def ticket_html(booking: dict, brand_name: str = "ديمو") -> str:
    """A printable ticket. Open it in a browser and choose Print > Save as PDF."""
    e = lambda v: html.escape(str(v))  # noqa: E731
    item = booking.get("item") or booking.get("item_name", "")
    nights = booking.get("nights") or 0
    rows = [("رقم الحجز", booking["ref"]), ("الاسم", booking["customer_name"]), ("الحجز", item),
            ("التاريخ", booking["start_date"]), ("عدد الأفراد", booking["people"])]
    if nights:
        rows.append(("عدد الليالي", nights))
    rows += [("الإجمالي", f"{int(booking['total_egp']):,} جنيه"), ("الحالة", booking.get("status", ""))]
    body = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in rows)
    return f"""<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>{e(brand_name)} | {e(booking['ref'])}</title>
<style>
body {{ font-family: Tahoma, Arial, sans-serif; background: #F4EDE4; color: #333333; padding: 32px; }}
.ticket {{ max-width: 560px; margin: auto; background: #fff; border-radius: 12px; overflow: hidden;
           box-shadow: 0 4px 12px rgba(26, 38, 57, 0.08); }}
.head {{ background: #1A2639; color: #D4AF37; padding: 22px 26px; }}
.head h1 {{ margin: 0; font-size: 30px; }} .head p {{ margin: 4px 0 0; color: #F4EDE4; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px 26px; text-align: right; border-bottom: 1px solid #E8DCC4; font-size: 16px; }}
th {{ color: #1A2639; width: 38%; }}
.ref {{ font-size: 22px; letter-spacing: 2px; direction: ltr; }}
.note {{ padding: 16px 26px; font-size: 13px; color: #6b655c; background: #F4EDE4; }}
@media print {{ body {{ background: #fff; padding: 0; }} }}
</style></head><body><div class="ticket">
<div class="head"><h1>{e(brand_name)}</h1><p>تذكرة حجز</p></div>
<table>{body}</table>
<div class="note">حجز تجريبي لأغراض المشروع: لا يوجد حجز فعلي أو دفع حقيقي. للطباعة أو الحفظ كملف PDF: اطبع الصفحة واختر "حفظ بتنسيق PDF".</div>
</div></body></html>"""
