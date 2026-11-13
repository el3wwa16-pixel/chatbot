"""Demo's visual identity ("Nile night"): colours, typefaces, logo and page styling.

Deep night blue with gold, like the Nile after dark. The full guide is in BRAND.md.
"""
import base64
import html
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
ICON = str(ASSETS / "icon.png")  # browser tab icon and the bot's chat avatar

NIGHT = "#0D1330"    # ليل: page background
DEEP = "#131A3A"     # كحلي: text on gold
CARD = "#1C2550"     # بطاقة: bot bubbles, sidebar
LINE = "#2C3870"     # حدود
GOLD = "#E9C46A"     # ذهب: name, buttons, the traveller's bubble
CREAM = "#F2EBDD"    # كريمي: body text
MIST = "#9FA8CC"     # ضباب: secondary text

_WAVE = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' "
         "height='12'><path d='M0 10 L7 2 L14 10 L21 2 L28 10' fill='none' "
         "stroke='%23E9C46A' stroke-opacity='.55' stroke-width='2' stroke-linejoin='round'/></svg>")


def _logo_uri() -> str:
    data = base64.b64encode((ASSETS / "logo.svg").read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


def css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@600;800&family=Tajawal:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stChatInput"] textarea, button, input {{
    font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif;
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Each paragraph picks its own direction: Arabic right-to-left, English left-to-right. */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stSidebar"] p, h1, h2, h3, [data-testid="stCaptionContainer"] {{
    unicode-bidi: plaintext; text-align: start;
}}
[data-testid="stChatInput"] textarea {{ unicode-bidi: plaintext; }}

/* header */
.demo-hero {{ direction: rtl; display: flex; align-items: center; gap: 18px; margin: 4px 0 10px; }}
.demo-hero img {{ width: 80px; height: 80px; flex: none; }}
.demo-name {{ font-family: 'Cairo', sans-serif; font-weight: 800; font-size: 62px;
              line-height: 1.15; color: {GOLD}; text-align: right; }}
.demo-tag {{ font-size: 18px; color: {MIST}; font-weight: 500; text-align: right; }}
.demo-wave {{ height: 12px; background: url("{_WAVE}") repeat-x center; margin: 6px 0 24px; }}

/* headings */
h2, h3 {{ color: {GOLD}; font-family: 'Cairo', sans-serif; }}
[data-testid="stSidebar"] h2 {{ font-size: 19px; border-bottom: 1px solid {LINE};
                                padding-bottom: 6px; margin-top: 10px; }}

/* the bot's bubble (right side, avatar on the right) */
[data-testid="stChatMessage"] {{
    flex-direction: row-reverse; background: {CARD}; border: 1px solid {LINE};
    border-radius: 22px 22px 6px 22px; padding: 14px 16px; margin-left: 8%;
}}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ line-height: 1.85; font-size: 17px; }}
[data-testid="stChatMessage"] a, [data-testid="stChatMessage"] strong {{ color: {GOLD}; }}

/* the traveller's bubble (left side, gold) */
.demo-me {{ display: flex; justify-content: flex-start; margin: 2px 0 14px; }}
.demo-me > div {{ background: {GOLD}; color: {DEEP}; font-weight: 500; font-size: 17px;
                  line-height: 1.7; padding: 12px 18px; border-radius: 22px 22px 22px 6px;
                  max-width: 82%; unicode-bidi: plaintext; text-align: start;
                  white-space: pre-wrap; overflow-wrap: anywhere; }}

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 999px; border: 1px solid {GOLD}; color: {GOLD};
    background: transparent; font-weight: 500; transition: all .15s ease;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    background: {GOLD}; color: {DEEP}; border-color: {GOLD};
}}
.stButton > button p {{ unicode-bidi: plaintext; }}

.demo-hint {{ direction: rtl; text-align: right; font-size: 15px; color: {MIST}; margin: 6px 2px; }}
</style>
"""


def hero_html(name: str, tagline: str) -> str:
    return (f'<div class="demo-hero"><img src="{_logo_uri()}" alt="">'
            f'<div><div class="demo-name">{name}</div>'
            f'<div class="demo-tag">{tagline}</div></div></div>'
            f'<div class="demo-wave"></div>')


def user_bubble(text: str) -> str:
    return f'<div class="demo-me"><div>{html.escape(text)}</div></div>'
