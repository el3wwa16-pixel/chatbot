"""External knowledge sources (free, no API key): Wikipedia (Arabic and English)
and the Wikivoyage travel guide. Used when the local knowledge base has no good
answer. Text from these sites is shown to the user as content, never followed as
instructions.
"""
import re

import requests

from .textutil import normalize

HEADERS = {"User-Agent": "demo-tourism-chatbot/1.0 (student project)"}

SOURCES = {
    "ar": [("ar.wikipedia.org", "ويكيبيديا العربية"), ("en.wikipedia.org", "Wikipedia")],
    "en": [("en.wikipedia.org", "Wikipedia"), ("en.wikivoyage.org", "Wikivoyage")],
}

# Words that carry no meaning for a search engine (compared after normalize()).
_FILLER = {normalize(w) for w in """
احكيلي احكي قولي قوللي قول كلمني عرفني عايز عاوز عايزه عاوزه اعرف أعرف اريد أريد ابغى
عن ايه إيه هو هي مين من ما ماذا هل ازاي إزاي كيف فين أين اين امتى إمتى متى كام ليه لماذا
معلومات معلومة نبذة حاجة ده دي دا يعني لو سمحت فضلك ممكن طيب بقى كده يا انا أنا اللي
tell me about what is are was were who whom the a an where when how why please can you
i want to know do does did of information some give explain
""".split()}

_WORD = re.compile(r"[^\W_]+")


def clean_query(text: str) -> str:
    """'احكيلي عن معبد دندرة' -> 'معبد دندرة'. Falls back to the original text."""
    kept = [w for w in _WORD.findall(text) if normalize(w) not in _FILLER]
    return " ".join(kept) or text.strip()


def _search_site(host: str, query: str) -> dict | None:
    data = requests.get(f"https://{host}/w/api.php", params={
        "action": "query", "format": "json", "redirects": 1,
        "generator": "search", "gsrsearch": query, "gsrlimit": 4, "gsrnamespace": 0,
        "prop": "extracts|info|pageprops", "exintro": 1, "explaintext": 1,
        "exchars": 900, "exlimit": 4, "inprop": "url", "ppprop": "disambiguation",
    }, headers=HEADERS, timeout=10).json()
    pages = (data.get("query") or {}).get("pages") or {}
    for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        extract = (page.get("extract") or "").strip()
        if len(extract) < 60 or "disambiguation" in (page.get("pageprops") or {}):
            continue
        return {"title": page.get("title", ""), "extract": extract,
                "url": page.get("fullurl", f"https://{host}/")}
    return None


def search_web_knowledge(query: str, lang: str = "ar") -> dict:
    """Look a topic up in Wikipedia / Wikivoyage. Returns title, extract, url, source."""
    q = clean_query(str(query or ""))
    if len(q) < 2:
        return {"error": "Empty query"}
    problems = []
    for host, label in SOURCES.get(lang if lang in SOURCES else "ar"):
        try:
            hit = _search_site(host, q)
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError) as e:
            problems.append(f"{host}: {type(e).__name__}")
            continue
        if hit:
            return {"source": label, "query": q, **hit}
    if problems:
        return {"error": "External sources are unavailable right now (needs internet).",
                "detail": "; ".join(problems)}
    return {"error": f"Nothing found for: {q}"}
