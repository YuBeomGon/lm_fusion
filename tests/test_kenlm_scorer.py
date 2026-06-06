import math, os, pytest
from bpe_lm_fusion.kenlm_scorer import KenlmScorer

LM = "data/lm_honest_5g.binary"
pytestmark = pytest.mark.skipif(not os.path.exists(LM), reason="train LM first (Task 5)")

def test_score_returns_natural_log_negative():
    s = KenlmScorer(LM)
    state = s.start_state()
    lp = s.token_logprob(state, 1234)   # ln-scale, 확률이므로 <= 0
    assert lp <= 0.0

def test_oov_token_low_score():
    s = KenlmScorer(LM)
    state = s.start_state()
    common = s.token_logprob(state, s.most_common_first_token())
    oov = s.token_logprob(state, 999999)   # 코퍼스에 없을 가능성 큰 id
    assert oov <= common

def test_state_from_history_matches_full_replay():
    """긴 history에 대해 최적화(last order-1) state == 전체 BeginSentence replay."""
    s = KenlmScorer(LM)
    hist = [100, 200, 300, 400, 500, 600, 700, 800]   # len > order-1
    full = s.start_state()
    for t in hist:
        _, full = s.advance(full, t)
    opt = s.state_from_history(hist)
    for tok in (100, 250, 900, 1234, 5000):
        assert abs(s.token_logprob(full, tok) - s.token_logprob(opt, tok)) < 1e-9

def test_state_from_history_short_uses_begin_sentence():
    """history가 order-1 이하면 <s> 문맥 유지(BeginSentence부터 replay)."""
    s = KenlmScorer(LM)
    hist = [100, 200]                                  # len <= order-1
    full = s.start_state()
    for t in hist:
        _, full = s.advance(full, t)
    opt = s.state_from_history(hist)
    for tok in (100, 300, 1234):
        assert abs(s.token_logprob(full, tok) - s.token_logprob(opt, tok)) < 1e-9
