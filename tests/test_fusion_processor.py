import torch
from bpe_lm_fusion.fusion_processor import BpeKenlmFusionProcessor

class FakeScorer:
    def state_from_history(self, ids): return ("S", tuple(ids))
    def token_logprob(self, state, tok):
        return 0.0 if tok == 7 else -10.0   # 토큰 7만 LM이 선호

def test_topk_boosts_lm_favored_token():
    proc = BpeKenlmFusionProcessor(
        scorer=FakeScorer(), alpha=1.0, asr_topk=3,
        skip_ids={0}, mode="topk")
    input_ids = torch.tensor([[0, 5, 6]])          # 0=special(skip)
    scores = torch.full((1, 10), -5.0)
    scores[0, 7] = -1.0; scores[0, 8] = -0.5; scores[0, 9] = -0.6  # top3 = 8,9,7
    out = proc(input_ids, scores.clone())
    # 7만 LM logprob 0 (나머지 -10) -> top-k(8,9,7)에서 7이 최상위로 올라옴
    assert out[0, 7] > out[0, 8]

def test_special_only_history_uses_start_state():
    proc = BpeKenlmFusionProcessor(
        scorer=FakeScorer(), alpha=1.0, asr_topk=2,
        skip_ids={0}, mode="topk")
    input_ids = torch.tensor([[0]])     # special만 -> 빈 history
    scores = torch.zeros((1, 10))
    out = proc(input_ids, scores)       # 예외 없이 동작
    assert out.shape == (1, 10)

def test_topk_keeps_neg_inf_masked():
    proc = BpeKenlmFusionProcessor(
        scorer=FakeScorer(), alpha=1.0, asr_topk=3,
        skip_ids=set(), mode="topk")
    input_ids = torch.tensor([[5, 6]])
    scores = torch.full((1, 10), -5.0)
    scores[0, 7] = float("-inf")          # token 7 suppressed
    scores[0, 8] = -0.5; scores[0, 9] = -0.6   # top3 = 8, 9, 7(-inf)
    out = proc(input_ids, scores.clone())
    assert out[0, 7] == float("-inf")     # stays dead despite LM favoring it

def test_full_vocab_boosts_lm_favored_token():
    proc = BpeKenlmFusionProcessor(
        scorer=FakeScorer(), alpha=1.0, asr_topk=3,
        skip_ids=set(), mode="full_vocab")
    input_ids = torch.tensor([[5, 6]])
    scores = torch.full((1, 10), -5.0)
    out = proc(input_ids, scores.clone())
    # full_vocab: 모든 토큰에 LM 가산. 7만 lm=0(나머지 -10) -> 7이 최댓값
    assert int(out[0].argmax()) == 7
