"""
bpe_lm_fusion/metrics.py
Eval metrics. CER/WER via jiwer; domain-term recall/precision; insertion.
"""
from __future__ import annotations
import jiwer


def cer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.cer(refs, hyps)


def wer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.wer(refs, hyps)


def _compact_spaces(text: str) -> str:
    return "".join(str(text).split())


def _unique_terms(terms, ignore_space: bool) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        key = _compact_spaces(term) if ignore_space else str(term)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return sorted(unique, key=len, reverse=True)


def _longest_match_counts(text: str, terms, ignore_space: bool = True) -> dict[str, int]:
    """Count non-overlapping term occurrences, preferring longer nested terms."""
    haystack = _compact_spaces(text) if ignore_space else str(text)
    occupied = [False] * len(haystack)
    counts: dict[str, int] = {}
    for term in _unique_terms(terms, ignore_space):
        start = 0
        term_len = len(term)
        while term_len and start < len(haystack):
            pos = haystack.find(term, start)
            if pos < 0:
                break
            end = pos + term_len
            if not any(occupied[pos:end]):
                for i in range(pos, end):
                    occupied[i] = True
                counts[term] = counts.get(term, 0) + 1
            start = pos + 1
    return counts


def term_recall_precision(refs, hyps, terms, ignore_space: bool = True) -> dict:
    """Occurrence-level metric with longest-match, non-overlapping term counts."""
    ref_total = hyp_total = matched = 0
    for ref, hyp in zip(refs, hyps):
        ref_counts = _longest_match_counts(ref, terms, ignore_space=ignore_space)
        hyp_counts = _longest_match_counts(hyp, terms, ignore_space=ignore_space)
        for term in set(ref_counts) | set(hyp_counts):
            rc, hc = ref_counts.get(term, 0), hyp_counts.get(term, 0)
            ref_total += rc
            hyp_total += hc
            matched += min(rc, hc)
    recall = matched / ref_total if ref_total else 0.0
    precision = matched / hyp_total if hyp_total else 0.0
    return {"recall": recall, "precision": precision,
            "ref_total": ref_total, "hyp_total": hyp_total, "matched": matched}


def insertion_rate(refs, hyps) -> float:
    """jiwer 정렬 기반 insertion / ref word count."""
    out = jiwer.process_words(refs, hyps)
    n_ref = sum(len(r) for r in out.references)
    return out.insertions / n_ref if n_ref else 0.0


def repeated_text_rate(hyps: list[str], n: int = 3, max_repeat: int = 3) -> float:
    """반복(환각) hyp 비율: 임의 word n-gram이 max_repeat 초과로 등장하면 flag."""
    if not hyps:
        return 0.0
    flagged = 0
    for hyp in hyps:
        words = hyp.split()
        if len(words) < n:
            continue
        counts: dict[tuple, int] = {}
        for i in range(len(words) - n + 1):
            g = tuple(words[i:i + n])
            counts[g] = counts.get(g, 0) + 1
        if any(c > max_repeat for c in counts.values()):
            flagged += 1
    return flagged / len(hyps)


def length_ratio_stats(refs: list[str], hyps: list[str], outlier: float = 2.0) -> dict:
    """hyp/ref 문자길이(공백제거) 비율 통계. ref 길이 0 쌍은 제외."""
    ratios = []
    for ref, hyp in zip(refs, hyps):
        rl = len("".join(ref.split()))
        if rl == 0:
            continue
        ratios.append(len("".join(hyp.split())) / rl)
    if not ratios:
        return {"mean_ratio": 0.0, "outlier_rate": 0.0}
    mean_ratio = sum(ratios) / len(ratios)
    outlier_rate = sum(1 for r in ratios if r > outlier) / len(ratios)
    return {"mean_ratio": mean_ratio, "outlier_rate": outlier_rate}


def no_speech_hallucination_rate(refs: list[str], hyps: list[str],
                                 ref_max_chars: int = 2, hyp_min_chars: int = 5) -> float:
    """ref이 사실상 무음(<=ref_max_chars)인데 hyp이 길게(>=hyp_min_chars) 나온 비율."""
    near_empty = halluc = 0
    for ref, hyp in zip(refs, hyps):
        if len("".join(ref.split())) <= ref_max_chars:
            near_empty += 1
            if len("".join(hyp.split())) >= hyp_min_chars:
                halluc += 1
    return halluc / near_empty if near_empty else 0.0


def hallucination_phrase_hit_rate(hyps: list[str], phrases: list[str]) -> float:
    """주어진 환각 phrase 중 하나라도 substring으로 포함한 hyp 비율."""
    if not hyps or not phrases:
        return 0.0
    hits = sum(1 for h in hyps if any(p in h for p in phrases))
    return hits / len(hyps)
