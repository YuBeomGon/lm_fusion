# bpe_lm_fusion/corpus.py
"""Build BPE token-id pseudo-word corpus for KenLM.

각 transcript -> Whisper token id -> "t<ID> t<ID> ..." 한 줄.
augmentation은 POC 범위 제외(추후 설계). build_corpus는 canonical only.
"""
from __future__ import annotations
from .normalize import normalize_text


def tokens_to_line(token_ids: list[int]) -> str:
    return " ".join(f"t{i}" for i in token_ids)


def text_to_canonical_line(text: str, tokenizer) -> str:
    text = normalize_text(text)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return tokens_to_line(ids)


def build_corpus(texts: list[str], tokenizer) -> list[str]:
    """Canonical corpus lines (one per text). Empty lines dropped."""
    lines = []
    for t in texts:
        line = text_to_canonical_line(t, tokenizer)
        if line:
            lines.append(line)
    return lines
