"""Pull an Authorization token out of pasted 'Copy as cURL / fetch' text."""
import re

_HEADER_PATTERNS = [
    re.compile(r"""["']?authorization["']?\s*[:=]\s*["']([^"'\r\n]+)["']""", re.I),
    re.compile(r"""-H\s+["']authorization:\s*([^"'\r\n]+)["']""", re.I),
    re.compile(r"""authorization:\s*([^\s"'\r\n]+)""", re.I),
]
_TOKEN_SHAPE = re.compile(r"(mfa\.[\w-]{20,}|[\w-]{20,30}\.[\w-]{5,10}\.[\w-]{20,50})")


def extract_token(raw: str) -> str:
    s = raw.strip()
    for pat in _HEADER_PATTERNS:
        m = pat.search(s)
        if m and m.group(1):
            return m.group(1).strip()
    m = _TOKEN_SHAPE.search(s)
    if m:
        return m.group(1).strip()
    return s.strip("\"'")
