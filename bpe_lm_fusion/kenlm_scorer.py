# bpe_lm_fusion/kenlm_scorer.py
"""KenLM wrapper for BPE token-id pseudo-words.

KenLM은 log10을 반환 -> ln(10) 곱해 natural log로 변환(Whisper log_softmax와 정합).
state는 token history로 재계산(stateless): beam reorder 회피. POC 한정.
"""
from __future__ import annotations
import math
import kenlm

LN10 = math.log(10.0)


def _word(token_id: int) -> str:
    return f"t{token_id}"


class KenlmScorer:
    def __init__(self, binary_path: str):
        self.model = kenlm.Model(binary_path)
        self.order = self.model.order          # n-gram order (e.g. 5)

    def start_state(self) -> "kenlm.State":
        s = kenlm.State()
        self.model.BeginSentenceWrite(s)   # python kenlm API (NOT BeginSentenceState)
        return s

    def null_state(self) -> "kenlm.State":
        s = kenlm.State()
        self.model.NullContextWrite(s)     # context-free start (no <s>)
        return s

    def advance(self, in_state, token_id: int):
        """Return (logprob_ln, out_state) after consuming token."""
        out = kenlm.State()
        lp10 = self.model.BaseScore(in_state, _word(token_id), out)
        return lp10 * LN10, out

    def state_from_history(self, token_ids):
        """State after a token-id history.

        n-gram(order N) state는 마지막 N-1 토큰만으로 완전히 결정된다(Markov).
        따라서 history가 길면 마지막 order-1 토큰만 NullContext에서 replay한다
        — 전체 replay와 수치적으로 동일하며 step당 O(L)→O(order) 로 줄인다.
        history가 order-1 이하면 <s> 문맥이 유효하므로 BeginSentence부터 replay.
        """
        n = self.order - 1
        if len(token_ids) <= n:
            s = self.start_state()
            ctx = token_ids
        else:
            s = self.null_state()
            ctx = token_ids[-n:]
        for t in ctx:
            _, s = self.advance(s, t)
        return s

    def token_logprob(self, in_state, token_id: int) -> float:
        """ln-scale logprob of token given state (state not mutated)."""
        out = kenlm.State()
        return self.model.BaseScore(in_state, _word(token_id), out) * LN10

    # 테스트 편의용
    def most_common_first_token(self) -> int:
        return 0
