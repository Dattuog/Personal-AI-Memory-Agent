import re


SENSITIVE_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",
    r"\b\d{13,19}\b",
    r"-----BEGIN.*PRIVATE KEY-----",
]


def looks_sensitive(content: str) -> bool:
    return any(re.search(pattern, content, flags=re.DOTALL) for pattern in SENSITIVE_PATTERNS)
