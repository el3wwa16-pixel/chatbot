"""Actions the model can call: weather, hotel/tour search, bookings, human hand-off.

Each function returns a plain dict. Errors come back as {"error": "..."} so the
model can explain the problem to the tourist instead of crashing.
"""
import math
import secrets
import string
from datetime import date, datetime

import requests

from . import db
from .cities import CITIES, canonical_city

WEATHER_CODES = {
    0: "سماء صافية", 1: "صافٍ غالبًا", 2: "غائم جزئيًا", 3: "غائم",
    45: "ضباب", 48: "ضباب", 51: "رذاذ خفيف", 53: "رذاذ", 55: "رذاذ كثيف",
    61: "مطر خفيف", 63: "مطر", 65: "مطر غزير", 80: "زخات مطر", 81: "زخات مطر",
    82: "زخات غزيرة", 95: "عاصفة رعدية", 96: "عاصفة رعدية", 99: "عاصفة رعدية",
}


def _int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ---- weather (Open-Meteo: free, no key) --------------------------------
def get_weather(city: str) -> dict:
    name = canonical_city(city)
    try:
        if name:
            lat, lon = CITIES[name]["lat"], CITIES[name]["lon"]
        else:
            g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": city, "count": 1}, timeout=10).json()
            if not g.get("results"):
                return {"error": f"City not found: {city}"}
            hit = g["results"][0]
            name, lat, lon = hit["name"], hit["latitude"], hit["longitude"]
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        }, timeout=10).json()
        cur, daily = w["current"], w["daily"]
        return {
            "city": name,
            "now": {"temperature_c": cur["temperature_2m"],
                    "condition": WEATHER_CODES.get(cur.get("weather_code"), "غير محدد"),
                    "wind_kmh": cur.get("wind_speed_10m")},
            "forecast": [
                {"date": d, "max_c": hi, "min_c": lo, "rain_chance_percent": rain}
                for d, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                                           daily["temperature_2m_min"],
                                           daily["precipitation_probability_max"])],
        }
    except (requests.RequestException, KeyError, ValueError):
        return {"error": "Weather service is unavailable right now (needs internet)."}


# ---- catalogue ---------------------------------------------------------
def search_hotels(db_path, city: str, max_price_egp=None, min_stars=None) -> dict:
    name = canonical_city(city)
    if not name:
        return {"error": f"Unknown city: {city}", "available_cities": list(CITIES)}
    sql, params = "SELECT * FROM hotels WHERE city = ?", [name]
    if _int(max_price_egp):
        sql += " AND price_per_night_egp <= ?"
        params.append(_int(max_price_egp))
    if _int(min_stars):
        sql += " AND stars >= ?"
        params.append(_int(min_stars))
    rows = db.query(db_path, sql + " ORDER BY price_per_night_egp", tuple(params))
    return {"city": name, "hotels": rows, "note": "Demo catalogue (fictional hotels)."}


def search_tours(db_path, city: str = "") -> dict:
    name = canonical_city(city)
    if city and not name:
        return {"error": f"Unknown city: {city}", "available_cities": list(CITIES)}
    if name:
        rows = db.query(db_path, "SELECT * FROM tours WHERE city = ? "
                                 "ORDER BY price_per_person_egp", (name,))
    else:
        rows = db.query(db_path, "SELECT * FROM tours ORDER BY city")
    return {"city": name or "all", "tours": rows, "note": "Demo catalogue (fictional tours)."}


# ---- bookings ----------------------------------------------------------
def _new_ref() -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "TB-" + "".join(secrets.choice(alphabet) for _ in range(6))


def create_booking(db_path, kind: str, item_id, customer_name: str, start_date: str,
                   nights=1, people=1) -> dict:
    kind = str(kind or "").strip().lower()
    if kind not in ("hotel", "tour"):
        return {"error": "kind must be 'hotel' or 'tour'"}
    if not str(customer_name or "").strip():
        return {"error": "customer_name is required; ask the customer for their name"}
    try:
        day = date.fromisoformat(str(start_date).strip()[:10])
    except ValueError:
        return {"error": "start_date must look like YYYY-MM-DD"}
    if day < date.today():
        return {"error": "start_date is in the past"}
    people, nights = _int(people, 1), _int(nights, 1)
    if not 1 <= people <= 20:
        return {"error": "people must be between 1 and 20"}
    if not 1 <= nights <= 30:
        return {"error": "nights must be between 1 and 30"}

    table = "hotels" if kind == "hotel" else "tours"
    rows = db.query(db_path, f"SELECT * FROM {table} WHERE id = ?", (_int(item_id, -1),))
    if not rows:
        return {"error": f"No {kind} with id {item_id}. Search first to get a valid id."}
    item = rows[0]
    if kind == "hotel":
        rooms = math.ceil(people / 2)
        total = item["price_per_night_egp"] * nights * rooms
    else:
        nights, rooms = 0, 0
        total = item["price_per_person_egp"] * people

    ref = _new_ref()
    db.execute(db_path, "INSERT INTO bookings VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
        ref, kind, item["id"], item["name"], customer_name.strip(), day.isoformat(),
        nights, people, total, "confirmed", datetime.now().isoformat(timespec="seconds")))
    out = {"status": "confirmed", "ref": ref, "kind": kind, "item": item["name"],
           "city": item["city"], "customer_name": customer_name.strip(),
           "start_date": day.isoformat(), "people": people, "total_egp": total,
           "note": "Demo booking: no real reservation or payment was made."}
    if kind == "hotel":
        out.update(nights=nights, rooms=rooms)
    return out


def get_booking(db_path, ref: str) -> dict:
    rows = db.query(db_path, "SELECT * FROM bookings WHERE ref = ?",
                    (str(ref or "").strip().upper(),))
    return rows[0] if rows else {"error": f"No booking found with reference {ref}"}


def cancel_booking(db_path, ref: str) -> dict:
    ref = str(ref or "").strip().upper()
    rows = db.query(db_path, "SELECT * FROM bookings WHERE ref = ?", (ref,))
    if not rows:
        return {"error": f"No booking found with reference {ref}"}
    if rows[0]["status"] == "cancelled":
        return {"error": "This booking is already cancelled"}
    db.execute(db_path, "UPDATE bookings SET status = 'cancelled' WHERE ref = ?", (ref,))
    return {"status": "cancelled", "ref": ref, "item": rows[0]["item_name"]}


def handoff_to_human(db_path, reason: str) -> dict:
    db.execute(db_path, "INSERT INTO handoffs (reason, created_at) VALUES (?, ?)",
               (str(reason or "").strip() or "unspecified",
                datetime.now().isoformat(timespec="seconds")))
    ticket = db.query(db_path, "SELECT MAX(id) AS id FROM handoffs")[0]["id"]
    return {"status": "ticket_opened", "ticket": ticket,
            "note": "Simulated hand-off: a human agent would contact the customer."}


# ---- registry ----------------------------------------------------------
def _spec(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SPECS = [
    _spec("get_weather", "Current weather and 3-day forecast for a city.",
          {"city": {"type": "string", "description": "City name, Arabic or English"}}, ["city"]),
    _spec("search_hotels", "List available hotels in a city, optionally filtered.",
          {"city": {"type": "string"},
           "max_price_egp": {"type": "integer", "description": "Max price per night in EGP"},
           "min_stars": {"type": "integer"}}, ["city"]),
    _spec("search_tours", "List bookable guided tours. Leave city empty for all cities.",
          {"city": {"type": "string"}}, []),
    _spec("create_booking",
          "Book a hotel or a tour. Only call after the customer confirmed the details. "
          "item_id must come from search_hotels or search_tours.",
          {"kind": {"type": "string", "enum": ["hotel", "tour"]},
           "item_id": {"type": "integer"},
           "customer_name": {"type": "string"},
           "start_date": {"type": "string", "description": "YYYY-MM-DD"},
           "nights": {"type": "integer", "description": "Hotels only"},
           "people": {"type": "integer"}},
          ["kind", "item_id", "customer_name", "start_date"]),
    _spec("get_booking", "Look up a booking by its reference (like TB-A1B2C3).",
          {"ref": {"type": "string"}}, ["ref"]),
    _spec("cancel_booking", "Cancel a booking by its reference.",
          {"ref": {"type": "string"}}, ["ref"]),
    _spec("handoff_to_human",
          "Open a ticket for a human agent: complaints, refunds, emergencies, "
          "or anything you cannot solve.",
          {"reason": {"type": "string"}}, ["reason"]),
]

_NEEDS_DB = {"search_hotels": search_hotels, "search_tours": search_tours,
             "create_booking": create_booking, "get_booking": get_booking,
             "cancel_booking": cancel_booking, "handoff_to_human": handoff_to_human}


def dispatch(name: str, args: dict, db_path) -> dict:
    """Run a tool by name. Never raises: bad calls come back as {'error': ...}."""
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be an object"}
    try:
        if name == "get_weather":
            return get_weather(**args)
        if name in _NEEDS_DB:
            return _NEEDS_DB[name](db_path, **args)
    except TypeError as e:
        return {"error": f"Wrong arguments for {name}: {e}"}
    return {"error": f"Unknown tool: {name}"}
