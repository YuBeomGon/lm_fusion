# bpe_lm_fusion/fusion_processor.py
"""LogitsProcessor: BPE-LM shallow fusion.

HF beam search는 logits processor 호출 전에 log_softmax를 적용하므로 `scores`는
이미 log-prob다. 따라서 추가 정규화 없이 더하기만 한다:
    scores[tok] += alpha * lm_logprob_ln
- 우리 프로세서는 기본 프로세서(SuppressTokens/ForceTokens/no-timestamps) 뒤에 실행된다.
  이미 -inf로 죽은 토큰은 건드리지 않고, special/timestamp 토큰(skip_ids)도 건너뛴다.
- mode=topk       -> ASR 상위 k 후보에만 LM 가산 (가이드 v2 방식 A)
- mode=full_vocab -> 전체 vocab에 LM 가산 (첫 token rescue/ceiling 진단; 느림 -> --limit 전용)
"""
from __future__ import annotations
import torch
from transformers import LogitsProcessor

_NEG_INF = float("-inf")


class BpeKenlmFusionProcessor(LogitsProcessor):
    def __init__(self, scorer, alpha: float, asr_topk: int,
                 skip_ids: set[int], mode: str = "topk"):
        super().__init__()
        self.scorer = scorer
        self.alpha = alpha
        self.asr_topk = asr_topk
        self.skip_ids = skip_ids   # special + timestamp ids: LM 미적용 & history 제외
        self.mode = mode

    def _history(self, row_ids: torch.Tensor) -> list[int]:
        return [int(t) for t in row_ids.tolist() if int(t) not in self.skip_ids]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return scores
        n_beam, vocab = scores.shape
        for b in range(n_beam):
            state = self.scorer.state_from_history(self._history(input_ids[b]))
            if self.mode == "full_vocab":
                cand = range(vocab)
            else:
                cand = torch.topk(scores[b], self.asr_topk).indices.tolist()
            for tok in cand:
                tok = int(tok)
                if tok in self.skip_ids:          # special/timestamp -> no LM
                    continue
                if scores[b, tok] == _NEG_INF:    # suppressed/forced-out -> keep dead
                    continue
                lm_lp = self.scorer.token_logprob(state, tok)
                scores[b, tok] = scores[b, tok] + self.alpha * lm_lp
        return scores
