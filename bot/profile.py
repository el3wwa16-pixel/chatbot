"""What the traveller told us about themselves in this chat (kept for the session only)."""
import re

_NAME = re.compile(r"(?:اسمي|إسمي|my name is|i am called|i'm called|call me)\s+([^\W\d_]{2,20})", re.I)
_KIDS = ("اطفال", "أطفال", "اولادي", "أولادي", "عيالي", "ولادي", "kids", "children", "my family")
_NOT_NAMES = {"ايه", "إيه", "what", "مين", "هو"}


def update(profile: dict, text: str) -> dict:
    """Reads a name or 'travelling with children' from the message. Returns what changed."""
    changed = {}
    match = _NAME.search(text)
    if match and match.group(1).lower() not in _NOT_NAMES:
        changed["name"] = match.group(1)
    if any(word in text.lower() for word in _KIDS) and not profile.get("kids"):
        changed["kids"] = True
    profile.update(changed)
    return changed


def describe(profile: dict) -> str:
    """One line for the model's instructions ('' when nothing is known)."""
    parts = []
    if profile.get("name"):
        parts.append(f"The traveller's name is {profile['name']}; use it now and then.")
    if profile.get("kids"):
        parts.append("They are travelling with children; prefer family-friendly suggestions.")
    return " ".join(parts)
