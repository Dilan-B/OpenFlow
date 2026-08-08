"""Deterministic repairs for things transcribers reliably get wrong.

Every fix here is one a person would make without thinking, so there is no
reason to spend a model round trip on it. The bar for inclusion is that the
input form is *unambiguous* -- if a correction needs context to be right, it
belongs to the LLM pass, not here.

That bar is why "its" is absent while "dont" is present: "its" is a real word
whose correction depends on meaning, but no English sentence wants "dont".
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Contractions the transcriber drops the apostrophe from.
# ---------------------------------------------------------------------------
# Only forms that are not themselves valid English words. "its", "were",
# "hell", "shed", "wed" and friends are deliberately missing: choosing between
# "its" and "it's" needs to know what the sentence means.
CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
    "cant": "can't", "couldnt": "couldn't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "isnt": "isn't", "arent": "aren't",
    "wasnt": "wasn't", "werent": "weren't", "hasnt": "hasn't",
    "havent": "haven't", "hadnt": "hadn't", "wont": "won't",
    "aint": "ain't", "mustnt": "mustn't", "neednt": "needn't",
    "im": "I'm", "ive": "I've", "youre": "you're", "youve": "you've",
    "youll": "you'll", "theyre": "they're", "theyve": "they've",
    "theyll": "they'll", "weve": "we've", "were": None,   # ambiguous, skip
    "whos": "who's", "whats": "what's", "wheres": "where's",
    "hows": "how's", "thats": "that's", "theres": "there's",
    "heres": "here's", "lets": None,                      # positional, below
    "oclock": "o'clock",
}
CONTRACTIONS = {k: v for k, v in CONTRACTIONS.items() if v}

# Valid words on their own, so only corrected in a position where the
# contraction is the only sensible reading.
SENTENCE_INITIAL_CONTRACTIONS = {"lets": "Let's"}

# ---------------------------------------------------------------------------
# Capitalization. Applied only when the transcriber produced an all-lowercase
# token, so anything it already cased correctly is left untouched.
# ---------------------------------------------------------------------------
ACRONYMS = {
    "api", "apis", "url", "urls", "sql", "html", "css", "json", "xml", "yaml",
    "http", "https", "ui", "ux", "ai", "ml", "llm", "cpu", "gpu", "ram", "ssd",
    "usb", "pdf", "csv", "faq", "ceo", "cto", "cfo", "hr", "kpi", "roi", "sdk",
    "ide", "cli", "gui", "os", "vpn", "dns", "ip", "ssh", "tls", "ssl", "jwt",
    "orm", "crud", "ci", "cd", "qa", "mvp", "saas", "rpc", "grpc", "uuid",
    "utc", "am", "pm", "usa", "uk", "eu",
}

PROPER_NOUNS = {
    "github": "GitHub", "gitlab": "GitLab", "javascript": "JavaScript",
    "typescript": "TypeScript", "python": "Python", "django": "Django",
    "postgres": "Postgres", "postgresql": "PostgreSQL", "mysql": "MySQL",
    "sqlite": "SQLite", "redis": "Redis", "mongodb": "MongoDB",
    "docker": "Docker", "kubernetes": "Kubernetes", "linux": "Linux",
    "ubuntu": "Ubuntu", "windows": "Windows", "macos": "macOS",
    "android": "Android", "iphone": "iPhone", "ipad": "iPad", "ios": "iOS",
    "youtube": "YouTube", "google": "Google", "gmail": "Gmail",
    "microsoft": "Microsoft", "apple": "Apple", "amazon": "Amazon",
    "netflix": "Netflix", "spotify": "Spotify", "slack": "Slack",
    "notion": "Notion", "figma": "Figma", "photoshop": "Photoshop",
    "openai": "OpenAI", "anthropic": "Anthropic", "claude": "Claude",
    "chatgpt": "ChatGPT", "gemini": "Gemini", "nvidia": "NVIDIA",
    "intel": "Intel", "react": "React", "angular": "Angular", "vue": "Vue",
    "node": "Node", "npm": "npm", "pytorch": "PyTorch", "numpy": "NumPy",
    "pandas": "pandas", "excel": "Excel", "powerpoint": "PowerPoint",
    "outlook": "Outlook", "zoom": "Zoom", "teams": "Teams",
    "salesforce": "Salesforce", "stripe": "Stripe", "shopify": "Shopify",
    "wifi": "Wi-Fi", "bluetooth": "Bluetooth", "englisch": "English",
    "english": "English", "spanish": "Spanish", "french": "French",
    "german": "German", "japanese": "Japanese", "chinese": "Chinese",
}

# ---------------------------------------------------------------------------
# Spoken symbols. Multi-word first so "at sign" beats a bare "at".
# ---------------------------------------------------------------------------
SPOKEN_SYMBOLS: tuple[tuple[str, str], ...] = (
    (r"\bat sign\b", "@"),
    (r"\bhash ?tag\b", "#"),
    (r"\bpound sign\b", "#"),
    (r"\bdollar sign\b", "$"),
    (r"\bpercent sign\b", "%"),
    (r"\bampersand\b", "&"),
    (r"\basterisk\b", "*"),
    (r"\bplus sign\b", "+"),
    (r"\bequals sign\b", "="),
    (r"\bforward slash\b", "/"),
    (r"\bback ?slash\b", "\\"),
    (r"\bopen paren(?:thesis)?\b", "("),
    (r"\bclose paren(?:thesis)?\b", ")"),
)

_WORD = r"[A-Za-z0-9_-]+"
# "john dot smith at gmail dot com" -> john.smith@gmail.com
_EMAIL_RE = re.compile(
    rf"\b({_WORD})((?:\s+dot\s+{_WORD})*)\s+at\s+({_WORD})\s+dot\s+"
    r"(com|net|org|io|ai|co|edu|gov|dev|app|me)\b",
    re.IGNORECASE,
)
# "example dot com" -> example.com
_DOMAIN_RE = re.compile(
    rf"\b({_WORD})\s+dot\s+(com|net|org|io|ai|co|edu|gov|dev|app)\b",
    re.IGNORECASE,
)
# "five thirty p m" -> the transcriber usually gives digits already: "5 30 p m"
_MERIDIEM_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?)\s*([ap])\.?\s*m\.?\b", re.IGNORECASE)
# "3 30 p m" -- the transcriber writes hour and minutes as separate numbers.
_SPLIT_TIME_RE = re.compile(r"\b(\d{1,2})\s+([0-5]\d)\s*([ap])\.?\s*m\.?\b",
                            re.IGNORECASE)
# Stutters: "the the report", "I I think"
_STUTTER_RE = re.compile(r"\b(\w+)(\s+\1)\b", re.IGNORECASE)
# Words English legitimately doubles.
STUTTER_EXCEPTIONS = {"had", "that", "very", "no", "ha", "bye", "so", "sh"}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")


def _fix_email(match: re.Match) -> str:
    local, extra, domain, tld = match.groups()
    local_parts = [local] + re.findall(_WORD, extra or "")
    # "dot" appears as a literal word in the captured run; drop it.
    local_parts = [p for p in local_parts if p.lower() != "dot"]
    # "meet me at gmail dot com" is a sentence, not an address.
    if len(local_parts) == 1 and _is_common(local_parts[0]):
        return match.group(0)
    return f"{'.'.join(local_parts)}@{domain.lower()}.{tld.lower()}"


def apply_word_fixes(text: str) -> tuple[str, int]:
    """Repairs that only touch words -- safe to run before the token pipeline.

    Nothing here introduces punctuation, so the sentence splitter downstream
    cannot be confused by the result.
    """
    if not text:
        return text, 0
    fixes = 0

    def _destutter(match: re.Match) -> str:
        word = match.group(1)
        return match.group(0) if word.lower() in STUTTER_EXCEPTIONS else word

    text, n = _STUTTER_RE.subn(_destutter, text)
    fixes += n

    def _token(match: re.Match) -> str:
        nonlocal fixes
        word = match.group(0)
        lower = word.lower()

        if lower in CONTRACTIONS and "'" not in word:
            fixed = CONTRACTIONS[lower]
            fixes += 1
            return fixed[0].upper() + fixed[1:] if word[0].isupper() else fixed

        # Casing only ever *adds* information, and only to a token the
        # transcriber left entirely lowercase -- never second-guess a term it
        # already cased, like "gRPC" or "iOS".
        if word.islower():
            if lower in PROPER_NOUNS and PROPER_NOUNS[lower] != word:
                fixes += 1
                return PROPER_NOUNS[lower]
            if lower in ACRONYMS:
                fixes += 1
                return word.upper()
        return word

    return _TOKEN_RE.sub(_token, text), fixes


def apply_symbol_fixes(text: str) -> tuple[str, int]:
    """Repairs that introduce punctuation -- must run *after* the token
    pipeline, which would otherwise read "gmail.com" as two sentences and
    re-space it into "Gmail. Com".
    """
    if not text:
        return text, 0
    fixes = 0

    text, n = _EMAIL_RE.subn(_fix_email, text)
    fixes += n
    text, n = _DOMAIN_RE.subn(
        lambda m: m.group(0) if _is_common(m.group(1))
        else f"{m.group(1).lower()}.{m.group(2).lower()}", text)
    fixes += n

    for pattern, symbol in SPOKEN_SYMBOLS:
        # Substitute through a callable: re treats a backslash in a template
        # replacement as an escape, so a literal "\" is a pattern error.
        text, n = re.subn(pattern, lambda _m, s=symbol: s, text, flags=re.IGNORECASE)
        fixes += n

    text, n = _SPLIT_TIME_RE.subn(
        lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3).upper()}M", text)
    fixes += n
    text, n = _MERIDIEM_RE.subn(lambda m: f"{m.group(1)} {m.group(2).upper()}M", text)
    fixes += n

    # Contractions that are real words elsewhere, so only corrected where the
    # contraction is the only sensible reading. Sentence boundaries are real
    # by this point, which is why this runs here and not with the word pass.
    def _initial(match: re.Match) -> str:
        nonlocal fixes
        prefix, word = match.group(1), match.group(2)
        replacement = SENTENCE_INITIAL_CONTRACTIONS.get(word.lower())
        if replacement is None:
            return match.group(0)
        fixes += 1
        return prefix + replacement

    text = re.sub(r"(^|[.!?]\s+)([A-Za-z]+)", _initial, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), fixes


def _is_common(word: str) -> bool:
    """Guards "meet me at gmail dot com" from becoming "me@gmail.com"."""
    from ..phonetics import COMMON_WORDS

    return word.lower() in COMMON_WORDS
