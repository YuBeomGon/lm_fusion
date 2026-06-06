"""Shared text normalization. Apply identically to LM corpus, hyp, and ref."""
import re

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = _NUM_COMMA.sub("", text)   # 28,150 -> 28150
    text = _WS.sub(" ", text)         # collapse all whitespace to single space
    return text.strip()
