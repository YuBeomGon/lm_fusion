# bpe_lm_fusion/oracle.py
"""Oracle diagnostics. n-best oracle term recall.

reference에 등장한 용어가 n-best 후보 중 '하나라도'에 있으면 recoverable.
이 값이 높은데 1-best recall이 낮으면 -> rerank/fusion 여지가 큼.
"""
from __future__ import annotations


def nbest_oracle_term_recall(refs, nbests, terms) -> dict:
    ref_total = recovered = 0
    for ref, nbest in zip(refs, nbests):
        joined = "\n".join(nbest)
        for term in terms:
            rc = ref.count(term)
            ref_total += rc
            if rc and term in joined:
                recovered += rc
    return {"recall": recovered / ref_total if ref_total else 0.0,
            "ref_total": ref_total, "recovered": recovered}
