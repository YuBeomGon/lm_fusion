from bpe_lm_fusion.normalize import normalize_text

def test_collapse_whitespace():
    assert normalize_text("보장  개시일\t이후") == "보장 개시일 이후"

def test_strip_amount_commas():
    assert normalize_text("28,150원 입니다") == "28150원 입니다"

def test_strip_edges():
    assert normalize_text("  안녕하세요  ") == "안녕하세요"
