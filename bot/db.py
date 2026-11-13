"""SQLite storage: demo hotels and tours, real bookings and hand-off tickets.

The hotels and tours are FICTIONAL demo data so the booking flow can be shown
end to end without a real reservation system.
"""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS hotels (
    id INTEGER PRIMARY KEY, name TEXT, city TEXT, stars INTEGER,
    price_per_night_egp INTEGER, description TEXT);
CREATE TABLE IF NOT EXISTS tours (
    id INTEGER PRIMARY KEY, name TEXT, city TEXT, duration_hours REAL,
    price_per_person_egp INTEGER, description TEXT);
CREATE TABLE IF NOT EXISTS bookings (
    ref TEXT PRIMARY KEY, kind TEXT, item_id INTEGER, item_name TEXT,
    customer_name TEXT, start_date TEXT, nights INTEGER, people INTEGER,
    total_egp INTEGER, status TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, reason TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, source TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, rating INTEGER, created_at TEXT);
"""

HOTELS = [
    # name, city, stars, price/night, description
    ("فندق نسمة النيل (Nile Breeze)", "Cairo", 4, 3200, "على كورنيش النيل، قريب من وسط البلد والمتحف المصري."),
    ("بيت الخان (Khan House)", "Cairo", 3, 1500, "فندق صغير في القاهرة التاريخية على مسافة مشي من خان الخليلي."),
    ("هرم فيو لودج (Pyramid View Lodge)", "Giza", 4, 3800, "غرف بإطلالة على الأهرامات، قريب من المتحف المصري الكبير."),
    ("فندق فنار البحر (Fanar Hotel)", "Alexandria", 4, 2900, "على الكورنيش في محطة الرمل، إطلالة على الميناء الشرقي."),
    ("بنسيون بحري (Bahary Pension)", "Alexandria", 2, 950, "إقامة اقتصادية قرب قلعة قايتباي وسوق السمك."),
    ("منتجع ستانلي باي (Stanley Bay Resort)", "Alexandria", 5, 5600, "فندق فاخر أمام كوبري ستانلي مع شاطئ خاص."),
    ("دار طيبة (Thebes House)", "Luxor", 4, 2600, "على البر الشرقي بين معبد الأقصر والكرنك، حمام سباحة."),
    ("نُزل البر الغربي (West Bank Guesthouse)", "Luxor", 3, 1300, "بيت ريفي هادئ قريب من وادي الملوك."),
    ("فندق جزيرة النوبة (Nubia Island Hotel)", "Aswan", 4, 3000, "على جزيرة وسط النيل، مركب مجاني للبر."),
    ("بيت كاتو (Kato Nubian House)", "Aswan", 3, 1400, "بيت نوبي ملوّن في غرب سهيل."),
    ("منتجع مرجان (Morgan Reef Resort)", "Hurghada", 5, 6200, "شامل كليًا، شاطئ خاص ومركز غوص."),
    ("فندق الدهار (El Dahar Inn)", "Hurghada", 3, 1700, "في البلد القديمة، قريب من المارينا."),
    ("منتجع خليج نعمة (Naama Coral Resort)", "Sharm El Sheikh", 5, 6800, "على خليج نعمة، شعاب مرجانية أمام الفندق."),
    ("كامب القمر (Moon Camp)", "Dahab", 2, 800, "كامب بسيط على البحر في منطقة المشربة."),
    ("إيكولودج شالي (Shali Ecolodge)", "Siwa", 3, 2200, "مبني بالكرشيف على الطراز السيوي، بدون ضوضاء."),
    ("لودج تونس (Tunis Village Lodge)", "Fayoum", 3, 1900, "في قرية تونس المطلة على بحيرة قارون."),
    ("منتجع أبو دباب باي (Dabbab Bay Resort)", "Marsa Alam", 4, 4500, "على خليج مشهور بالسلاحف البحرية."),
]

TOURS = [
    # name, city, hours, price/person, description
    ("جولة الأهرامات وأبو الهول", "Giza", 4, 1200, "مع مرشد: هضبة الأهرامات، البانوراما، أبو الهول."),
    ("المتحف المصري الكبير مع مرشد", "Giza", 4, 1500, "جولة في قاعات المتحف ومجموعة توت عنخ آمون."),
    ("القاهرة الإسلامية مشيًا", "Cairo", 5, 900, "شارع المعز، الأزهر، الغورية، خان الخليلي."),
    ("عشاء على مركب نيلي", "Cairo", 2.5, 1100, "رحلة مسائية في النيل مع عشاء وعرض تنورة."),
    ("إسكندرية في يوم", "Alexandria", 7, 1300, "قلعة قايتباي، مكتبة الإسكندرية، المسرح الروماني، كوم الشقافة."),
    ("جولة أكل سكندري", "Alexandria", 3, 700, "كبدة إسكندراني، سمك في بحري، وحلويات وسط البلد."),
    ("البر الغربي: وادي الملوك وحتشبسوت", "Luxor", 6, 1600, "وادي الملوك، معبد حتشبسوت، تمثالا ممنون."),
    ("رحلة البالون الطائر", "Luxor", 3, 2800, "تحليق مع شروق الشمس فوق البر الغربي."),
    ("رحلة أبو سمبل من أسوان", "Aswan", 10, 2400, "انطلاق قبل الفجر بسيارة مكيفة مع مرشد."),
    ("معبد فيلة والقرية النوبية", "Aswan", 5, 1100, "مركب لمعبد فيلة ثم غرب سهيل."),
    ("سنوركلنج في جزيرة الجفتون", "Hurghada", 7, 1000, "مركب، غداء، ووقفتان للسنوركلنج."),
    ("محمية رأس محمد بالمركب", "Sharm El Sheikh", 7, 1400, "سنوركلنج في أشهر مواقع الشعاب المرجانية."),
    ("جبل موسى ودير سانت كاترين", "Dahab", 14, 1800, "صعود ليلي لمشاهدة الشروق ثم زيارة الدير."),
    ("سفاري بحر الرمال الأعظم", "Siwa", 5, 1300, "سيارات دفع رباعي، عين مياه ساخنة، وغروب في الصحراء."),
    ("وادي الحيتان ووادي الريان", "Fayoum", 8, 1200, "الشلالات، متحف الحفريات، وهياكل الحيتان."),
    ("دولفين ستايا", "Marsa Alam", 8, 1900, "رحلة بحرية لشعاب ستايا للسباحة قرب الدلافين."),
]


def connect(db_path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path) -> None:
    """Create tables and insert the demo catalogue once."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM hotels").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO hotels (name, city, stars, price_per_night_egp, description) "
                "VALUES (?,?,?,?,?)", HOTELS)
        if conn.execute("SELECT COUNT(*) FROM tours").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO tours (name, city, duration_hours, price_per_person_egp, description) "
                "VALUES (?,?,?,?,?)", TOURS)


def query(db_path, sql: str, params: tuple = ()) -> list[dict]:
    with connect(db_path) as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def execute(db_path, sql: str, params: tuple = ()) -> int:
    with connect(db_path) as conn:
        return conn.execute(sql, params).rowcount


def list_bookings(db_path) -> list[dict]:
    return query(db_path, "SELECT * FROM bookings ORDER BY created_at DESC")


def list_handoffs(db_path) -> list[dict]:
    return query(db_path, "SELECT * FROM handoffs ORDER BY id DESC")


# ---- usage statistics --------------------------------------------------------
SOURCES = ("faq", "knowledge", "web", "tool", "model", "none")


def log_question(db_path, text: str, source: str) -> None:
    """source: where the answer came from (one of SOURCES); 'none' = no answer found."""
    from datetime import datetime
    execute(db_path, "INSERT INTO questions (text, source, created_at) VALUES (?, ?, ?)",
            (str(text)[:300], source if source in SOURCES else "none",
             datetime.now().isoformat(timespec="seconds")))


def log_feedback(db_path, question: str, rating: int) -> None:
    """rating: 1 = helpful, 0 = not helpful."""
    from datetime import datetime
    execute(db_path, "INSERT INTO feedback (question, rating, created_at) VALUES (?, ?, ?)",
            (str(question)[:300], 1 if rating else 0, datetime.now().isoformat(timespec="seconds")))


def stats(db_path) -> dict:
    one = lambda sql: query(db_path, sql)[0]["n"]  # noqa: E731
    return {
        "questions": one("SELECT COUNT(*) AS n FROM questions"),
        "unanswered": one("SELECT COUNT(*) AS n FROM questions WHERE source = 'none'"),
        "bookings": one("SELECT COUNT(*) AS n FROM bookings"),
        "cancelled": one("SELECT COUNT(*) AS n FROM bookings WHERE status = 'cancelled'"),
        "handoffs": one("SELECT COUNT(*) AS n FROM handoffs"),
        "likes": one("SELECT COUNT(*) AS n FROM feedback WHERE rating = 1"),
        "dislikes": one("SELECT COUNT(*) AS n FROM feedback WHERE rating = 0"),
        "by_source": {r["source"]: r["n"] for r in query(
            db_path, "SELECT source, COUNT(*) AS n FROM questions GROUP BY source")},
        "top_questions": query(
            db_path, "SELECT text, COUNT(*) AS n FROM questions GROUP BY text ORDER BY n DESC, text LIMIT 10"),
        "unanswered_list": query(
            db_path, "SELECT text, COUNT(*) AS n FROM questions WHERE source = 'none' "
                     "GROUP BY text ORDER BY n DESC, text LIMIT 10"),
        "disliked_list": query(
            db_path, "SELECT question AS text, COUNT(*) AS n FROM feedback WHERE rating = 0 "
                     "GROUP BY question ORDER BY n DESC LIMIT 10"),
    }
