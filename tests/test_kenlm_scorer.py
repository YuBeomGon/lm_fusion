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
