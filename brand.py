"""Rafiq's visual identity: colours, typefaces, logo and the page styling.

Concept: the colours of ancient Egyptian jewellery (lapis lazuli, turquoise faience,
carnelian and gold) on limestone, with everyday Egyptian Ruqaa lettering for the name.
The full guide is in BRAND.md.
"""
import base64
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
ICON = str(ASSETS / "icon.png")       # browser tab icon and the bot's chat avatar
USER_ICON = str(ASSETS / "user.png")  # the traveller's chat avatar

LAPIS = "#1B2B6B"        # لازورد: name, headings
TURQUOISE = "#0B7A75"    # فيروز: buttons, links (white text passes contrast on it)
FAIENCE = "#3CC4BC"      # فيروز فاتح: decoration only
CARNELIAN = "#C8452C"    # عقيق: small highlights, warnings
GOLD = "#D9A441"         # ذهب: the star in the logo
LIMESTONE = "#F7F2E8"    # حجر جيري: page background
PAPYRUS = "#EDE4D3"      # بردي: sidebar, cards
INK = "#141B33"          # حبر: body text

_WAVE = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' "
         "height='12'><path d='M0 10 L7 2 L14 10 L21 2 L28 10' fill='none' "
         "stroke='%233CC4BC' stroke-width='2.5' stroke-linejoin='round'/></svg>")


def _logo_uri() -> str:
    data = base64.b64encode((ASSETS / "logo.svg").read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


def css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Aref+Ruqaa:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stChatInput"] textarea, button, input {{
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
}}

/* Each paragraph picks its own direction: Arabic right-to-left, English left-to-right. */
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
[data-testid="stSidebar"] p, h1, h2, h3, [data-testid="stCaptionContainer"] {{
    unicode-bidi: plaintext; text-align: start;
}}
[data-testid="stChatInput"] textarea {{ unicode-bidi: plaintext; }}

/* header */
.rafiq-hero {{ direction: rtl; display: flex; align-items: center; gap: 18px; margin: 4px 0 10px; }}
.rafiq-hero img {{ width: 84px; height: 84px; flex: none; }}
.rafiq-name {{ font-family: 'Aref Ruqaa', serif; font-weight: 700; font-size: 68px;
               line-height: 1.15; color: {LAPIS}; text-align: right; }}
.rafiq-tag {{ font-size: 17px; color: {TURQUOISE}; font-weight: 500; margin-top: 2px;
              text-align: right; }}
.rafiq-wave {{ height: 12px; background: url("{_WAVE}") repeat-x center; margin: 4px 0 22px; }}

/* headings */
h2, h3 {{ color: {LAPIS}; }}
[data-testid="stSidebar"] h2 {{ font-size: 19px; border-bottom: 2px solid {FAIENCE};
                                padding-bottom: 6px; margin-top: 10px; }}

/* chat */
[data-testid="stChatMessage"] {{ border-radius: 18px; padding: 14px 16px;
                                 border: 1px solid #E2D7C1; background: #FFFDF8; }}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ line-height: 1.85; }}
[data-testid="stChatMessage"] a {{ color: {TURQUOISE}; font-weight: 600; }}
[data-testid="stChatMessage"] strong {{ color: {LAPIS}; }}
[data-testid="stChatInput"] {{ border-radius: 999px; }}

/* buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {{
    border-radius: 999px; border: 1.5px solid {TURQUOISE}; color: {TURQUOISE};
    background: transparent; font-weight: 500; transition: all .15s ease;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    background: {TURQUOISE}; color: #FFFFFF; border-color: {TURQUOISE};
}}
.stButton > button p {{ unicode-bidi: plaintext; }}

.rafiq-hint {{ direction: rtl; text-align: right; font-size: 14px; color: #5B6379; margin: 6px 2px; }}
</style>
"""


def hero_html(name: str, tagline: str) -> str:
    return (f'<div class="rafiq-hero"><img src="{_logo_uri()}" alt="">'
            f'<div><div class="rafiq-name">{name}</div>'
            f'<div class="rafiq-tag">{tagline}</div></div></div>'
            f'<div class="rafiq-wave"></div>')
