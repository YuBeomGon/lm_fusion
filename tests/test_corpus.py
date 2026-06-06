from bpe_lm_fusion.corpus import tokens_to_line, text_to_canonical_line

class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        # 공백 기준 단어 -> 길이만큼 가짜 id
        ids = [100 + len(w) for w in text.split()]
        class R: pass
        r = R(); r.input_ids = ids; return r

def test_tokens_to_line():
    assert tokens_to_line([1234, 5678, 9012]) == "t1234 t5678 t9012"

def test_text_to_canonical_line():
    line = text_to_canonical_line("가 나 다", FakeTok())
    assert line == "t101 t101 t101"
