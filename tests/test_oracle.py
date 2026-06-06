from bpe_lm_fusion.oracle import nbest_oracle_term_recall

def test_oracle_recovers_if_any_hyp_has_term():
    refs = ["보장개시일 안내"]
    nbests = [["보장 개시일 안내", "보장개시일 안내", "보장개일 안내"]]  # 2번째에 정답 용어
    r = nbest_oracle_term_recall(refs, nbests, ["보장개시일"])
    assert r["recall"] == 1.0

def test_oracle_miss():
    refs = ["보장개시일 안내"]
    nbests = [["보장 개시일", "보장개일"]]
    r = nbest_oracle_term_recall(refs, nbests, ["보장개시일"])
    assert r["recall"] == 0.0
