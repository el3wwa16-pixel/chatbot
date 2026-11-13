"""Visual identity, built from the project's brand guidelines (Pharaonic / Greek style).

Colours, typefaces, bubble shapes, shadows and motion follow the guide. BRAND.md
lists every value and where it is used.
"""
import base64
import html
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
ICON = str(ASSETS / "icon.png")  # Eye of Horus: browser tab icon and the bot's avatar

NIGHT = "#1A2639"      # أزرق ليلي (Primary): header, user bubble, bot text
GOLD = "#D4AF37"       # ذهبي فرعوني (Accent): name, icons, button outlines
PAPYRUS = "#F4EDE4"    # ورق البردي (Background)
CHARCOAL = "#333333"   # فحمي داكن (Text)
BOT = "#E8DCC4"        # خلفية رسالة البوت
RADIUS = "12px"
SHADOW = "0 4px 12px rgba(26, 38, 57, 0.08)"

# Greek key (meander), drawn in night blue at 5% opacity and repeated as the page background.
_MEANDER = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='48' "
            "height='48'><path d='M0 40 H40 V8 H8 V32 H32 V16 H16 V24 H24' fill='none' "
            "stroke='%231A2639' stroke-opacity='.05' stroke-width='3'/></svg>")
# The same key in gold, as the border under the header.
_MEANDER_GOLD = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' "
                 "height='14'><path d='M0 12 H20 V2 H6 V8 H14' fill='none' "
                 "stroke='%23D4AF37' stroke-width='2'/></svg>")


def _logo_uri() -> str:
    data = base64.b64encode((ASSETS / "logo.svg").read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


def css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@600;800&family=Cinzel:wght@600;700&family=Inter:wght@400;500;600&family=Tajawal:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stChatInput"] textarea, button, input {{
    font-family: 'Tajawal', 'Inter', 'Segoe UI', Tahoma, sans-serif;
}}
[data-testid="stAppViewContainer"] {{
    background-color: {PAPYRUS}; background-image: url("{_MEANDER}");
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Each paragraph picks its own direction: Arabic right-to-left, English left-to-right. */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stSidebar"] p, h1, h2, h3, [data-testid="stCaptionContainer"] {{
    unicode-bidi: plaintext; text-align: start;
}}
[data-testid="stChatInput"] textarea {{ unicode-bidi: plaintext; }}

/* header: night-blue banner with the gold name and a gold meander border */
.nb-hero {{ direction: rtl; display: flex; align-items: center; gap: 16px;
            background: {NIGHT}; border-radius: {RADIUS}; padding: 18px 22px 26px;
            box-shadow: {SHADOW}; position: relative; overflow: hidden; margin: 0 0 22px; }}
.nb-hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 6px; height: 14px;
                   background: url("{_MEANDER_GOLD}") repeat-x center; opacity: .8; }}
.nb-hero img {{ width: 72px; height: 72px; flex: none; border-radius: 50%;
                border: 2px solid {GOLD}; }}
.nb-name {{ font-family: 'Cairo', 'Cinzel', serif; font-weight: 800; font-size: 50px;
            line-height: 1.2; color: {GOLD}; text-align: right; }}
.nb-sub {{ font-family: 'Cinzel', serif; font-weight: 600; font-size: 13px; letter-spacing: 3px;
           color: {GOLD}; opacity: .85; text-align: right; }}
.nb-tag {{ font-size: 16px; color: {PAPYRUS}; opacity: .9; text-align: right; margin-top: 4px; }}

/* headings */
h2, h3 {{ color: {NIGHT}; font-family: 'Cairo', 'Cinzel', serif; }}
[data-testid="stSidebar"] h2 {{ font-size: 19px; border-bottom: 2px solid {GOLD};
                                padding-bottom: 6px; margin-top: 10px; }}

@keyframes nb-in {{ from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }} }}

/* bot bubble */
[data-testid="stChatMessage"] {{
    flex-direction: row-reverse; background: {BOT}; border-radius: {RADIUS};
    padding: 14px 16px; width: 92%; margin-left: 8%; box-sizing: border-box;
    box-shadow: {SHADOW}; animation: nb-in .45s ease both;
}}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{
    color: {NIGHT}; line-height: 1.85; font-size: 16px; }}
[data-testid="stChatMessage"] strong {{ color: {NIGHT}; }}
[data-testid="stChatMessage"] a {{ color: {NIGHT}; font-weight: 700; text-decoration: underline;
                                   text-decoration-color: {GOLD}; }}

/* user bubble */
.nb-me {{ display: flex; justify-content: flex-start; margin: 2px 0 14px;
          animation: nb-in .45s ease both; }}
.nb-me > div {{ background: {NIGHT}; color: #FFFFFF; font-size: 16px; line-height: 1.7;
                padding: 12px 18px; border-radius: {RADIUS}; max-width: 82%;
                box-shadow: {SHADOW}; unicode-bidi: plaintext; text-align: start;
                white-space: pre-wrap; overflow-wrap: anywhere; }}

/* buttons: thin rounded gold outline, night blue on hover */
.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 999px; border: 1.5px solid {GOLD}; color: {NIGHT};
    background: rgba(255, 255, 255, .55); font-weight: 500; transition: all .2s ease;
    box-shadow: {SHADOW};
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    background: {NIGHT}; color: {GOLD}; border-color: {NIGHT};
}}
.stButton > button p {{ unicode-bidi: plaintext; }}

/* chat input and cards */
[data-testid="stChatInput"] {{ border: 1.5px solid {GOLD}; border-radius: {RADIUS};
                               box-shadow: {SHADOW}; }}
[data-testid="stExpander"] details {{ border-radius: {RADIUS}; }}

.nb-hint {{ direction: rtl; text-align: right; font-size: 15px; color: {CHARCOAL};
            opacity: .75; margin: 6px 2px; }}
</style>
"""


def hero_html(name: str, tagline: str, latin: str = "") -> str:
    sub = f'<div class="nb-sub">{html.escape(latin)}</div>' if latin else ""
    return (f'<div class="nb-hero"><img src="{_logo_uri()}" alt="">'
            f'<div><div class="nb-name">{html.escape(name)}</div>{sub}'
            f'<div class="nb-tag">{html.escape(tagline)}</div></div></div>')


def user_bubble(text: str) -> str:
    return f'<div class="nb-me"><div>{html.escape(text)}</div></div>'
