from bpe_lm_fusion.metrics import (
    cer, term_recall_precision, insertion_rate,
    repeated_text_rate, length_ratio_stats,
    no_speech_hallucination_rate, hallucination_phrase_hit_rate,
)

def test_cer_perfect():
    assert cer(["가나다"], ["가나다"]) == 0.0

def test_cer_one_sub():
    assert abs(cer(["가나다"], ["가라다"]) - 1/3) < 1e-9

def test_term_recall():
    refs = ["보장개시일 이후", "후유장해 발생"]
    hyps = ["보장개시일 이후", "후유 장해 발생"]   # 공백 차이는 동일 용어로 처리
    r = term_recall_precision(refs, hyps, ["보장개시일", "후유장해"])
    assert r["recall"] == 1.0

def test_term_recall_uses_longest_non_overlapping_match():
    refs = ["보험금지급사유 안내"]
    hyps = ["보험금 안내"]
    r = term_recall_precision(refs, hyps, ["보험금지급사유", "보험금"])
    assert r["ref_total"] == 1
    assert r["hyp_total"] == 1
    assert r["matched"] == 0
    assert r["recall"] == 0.0
    assert r["precision"] == 0.0

def test_insertion_rate_zero():
    assert insertion_rate(["가 나 다"], ["가 나 다"]) == 0.0

# --- repeated_text_rate ---
def test_repeated_text_rate_flags():
    # "가 나 다" 3-gram repeated 4 times (> max_repeat=3) → flagged
    hyps = ["가 나 다 가 나 다 가 나 다 가 나 다", "정상 문장 입니다"]
    assert repeated_text_rate(hyps, n=3, max_repeat=3) == 0.5

def test_repeated_text_rate_empty():
    assert repeated_text_rate([], n=3, max_repeat=3) == 0.0

def test_repeated_text_rate_too_short():
    # fewer than n words can't be flagged
    assert repeated_text_rate(["가 나"], n=3, max_repeat=3) == 0.0

# --- length_ratio_stats ---
def test_length_ratio_stats_basic():
    refs = ["가나다", "가나"]      # 3 chars, 2 chars
    hyps = ["가나다라라라", "가나"]  # 6 chars (ratio 2.0), 2 chars (ratio 1.0)
    r = length_ratio_stats(refs, hyps, outlier=2.0)
    assert abs(r["mean_ratio"] - 1.5) < 1e-9
    assert r["outlier_rate"] == 0.0  # 2.0 is not > 2.0

def test_length_ratio_stats_skip_empty_ref():
    refs = ["", "가나"]
    hyps = ["환각텍스트", "가나"]
    r = length_ratio_stats(refs, hyps, outlier=2.0)
    assert abs(r["mean_ratio"] - 1.0) < 1e-9  # only second pair counts
    assert r["outlier_rate"] == 0.0

# --- no_speech_hallucination_rate ---
def test_no_speech_hallucination_rate_basic():
    refs = ["", "가"]                 # both near-empty (<=2 chars)
    hyps = ["환각된긴문장", "음"]      # first >=5 chars → hallucination, second not
    assert no_speech_hallucination_rate(refs, hyps) == 0.5

def test_no_speech_hallucination_rate_no_empty_ref():
    refs = ["정상참조문장"]
    hyps = ["환각된긴문장"]
    assert no_speech_hallucination_rate(refs, hyps) == 0.0

# --- hallucination_phrase_hit_rate ---
def test_hallucination_phrase_hit_rate_basic():
    hyps = ["시청해주셔서 감사합니다", "정상 문장"]
    assert hallucination_phrase_hit_rate(hyps, ["감사합니다"]) == 0.5

def test_hallucination_phrase_hit_rate_empty():
    assert hallucination_phrase_hit_rate([], ["감사합니다"]) == 0.0
    assert hallucination_phrase_hit_rate(["감사합니다"], []) == 0.0
