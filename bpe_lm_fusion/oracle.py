"""
bpe_lm_fusion/oracle.py
Oracle diagnostics. n-best oracle term recall.

reference에 등장한 용어가 n-best 후보 중 '하나라도'에 있으면 recoverable.
이 값이 높은데 1-best recall이 낮으면 -> rerank/fusion 여지가 큼.
"""
from __future__ import annotations
from .metrics import _longest_match_counts


def nbest_oracle_term_recall(refs, nbests, terms, ignore_space: bool = True) -> dict:
    ref_total = recovered = 0
    for ref, nbest in zip(refs, nbests):
        ref_counts = _longest_match_counts(ref, terms, ignore_space=ignore_space)
        for term, rc in ref_counts.items():
            ref_total += rc
            if rc and any(
                _longest_match_counts(hyp, [term], ignore_space=ignore_space).get(term, 0)
                for hyp in nbest
            ):
                recovered += rc
    return {"recall": recovered / ref_total if ref_total else 0.0,
            "ref_total": ref_total, "recovered": recovered}
