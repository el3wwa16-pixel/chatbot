"""Arabic/English text normalisation and tokenising for keyword search."""
import re

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_ALEF = re.compile("[إأآا]")
_WORD = re.compile(r"[^\W_]+")

STOPWORDS = {
    # Arabic (already normalised)
    "في", "من", "علي", "الي", "عن", "ما", "هو", "هي", "ايه", "ازاي", "فين", "هل",
    "عايز", "عاوز", "عايزه", "عاوزه", "انا", "او", "مع", "لو", "ده", "دي", "اللي",
    "كل", "هناك", "هذا", "هذه", "التي", "الذي", "ان", "كان", "يا", "بس", "كده",
    "ممكن", "ابغي", "اريد", "ماذا", "اين", "كيف", "متي", "ولا", "عن", "حاجه",
    # English
    "the", "a", "an", "of", "in", "to", "is", "are", "what", "where", "how",
    "and", "or", "for", "i", "me", "my", "want", "can", "you", "do", "it", "on",
    "at", "about", "tell", "there", "with", "be", "s",
}


def normalize(text: str) -> str:
    """Lowercase and unify Arabic letter variants so 'الإسكندرية' == 'الاسكندريه'."""
    t = str(text).lower()
    t = _DIACRITICS.sub("", t)
    t = _ALEF.sub("ا", t)
    return (t.replace("ى", "ي").replace("ة", "ه")
             .replace("ؤ", "و").replace("ئ", "ي"))


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _WORD.findall(normalize(text)):
        if len(tok) > 4 and tok.startswith("ال"):
            tok = tok[2:]
        elif len(tok) > 5 and tok[0] in "وبلف" and tok[1:3] == "ال":
            tok = tok[3:]
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens
