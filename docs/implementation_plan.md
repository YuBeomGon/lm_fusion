# BPE Token-Level KenLM 1-Pass Fusion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Whisper 디코딩 중 도메인 텍스트로 만든 BPE token-level KenLM을 1-pass shallow fusion으로 결합해, 보험 도메인 용어 인식을 개선할 수 있는지 HuggingFace에서 먼저 검증한다.

**Architecture:** 두 트랙으로 분리한다. **Track 1 (HF POC)** = 효과 검증 — 순수 Python, `LogitsProcessor`로 매 step KenLM 점수를 logits에 가산. **Track 2 (CT2 serving)** = 서빙 — HF 효과 게이트 통과 시에만 착수하는 **별도 상세 플랜**(C++ 포크). 이 문서는 Track 1을 태스크 단위로 상세화하고, Track 2는 게이트·범위만 정의한다.

**Tech Stack:** Python 3.12, `transformers`(Whisper), `torch`, `datasets`, KenLM(toolkit `lmplz`/`build_binary` + python `kenlm`), `jiwer`, `pyyaml`, `numpy`, `pytest`.

**참조 문서:**
- 설계: [`bpe_kenlm_1pass_fusion_guide_v2.md`](./bpe_kenlm_1pass_fusion_guide_v2.md)
- 분석/배경: [`assets/Fusion Analysis ...`](./assets/)
- 데이터: [`dataset_description.md`](./dataset_description.md)

---

## 핵심 설계 결정 (실행 전 합의된 사항)

1. **POC 스코어러는 python `kenlm`** (C++ 포크는 Track 2). state는 매 step **history로 재계산(stateless)** — beam reorder 회피.
2. **토크나이저 통일**: 코퍼스 생성·디코딩 모두 `WhisperTokenizer`. token id `t<ID>` pseudo-word로 KenLM 학습.
3. **결합 방식**: HF beam search는 logits processor 호출 **전에 이미 `log_softmax`를 적용**하므로, 프로세서가 받는 `scores`는 이미 log-prob다. 따라서 추가 log_softmax 없이 **`scores[tok] += α·(KenLM_log10 × ln10)`** 로 더하기만 한다. KenLM은 log10 반환 → ln 변환 필수. 우리 프로세서는 기본 프로세서(SuppressTokens/ForceTokens/no-timestamps) **뒤에** 실행되므로 `-inf` 마스크 토큰과 special/timestamp 토큰은 건너뛴다.
4. **OOV 정책**: 도메인 코퍼스에 없는 토큰은 KenLM `<unk>` 점수(낮음). α가 작고 음향이 강하면 음향이 주도 → 별도 floor/interpolation 없이 진행, 일반 set CER로 부작용 감시.
5. **누수 조건 2종 병행**:
   - **honest** = KenLM corpus에 train text만 (보고용 성능)
   - **ceiling** = KenLM corpus에 train + test text (상한선/진단용, 성능값으로 보고 금지)
6. **소규모 코퍼스**: `lmplz`에 `--discount_fallback` 필수.
7. **효과는 HF, latency는 CT2**: HF에서 속도 측정 안 함.
8. **BPE augmentation은 POC 범위에서 제외 (추후 별도 설계)**: HF `WhisperTokenizer`가 deterministic이라 stochastic BPE dropout을 직접 못 한다. POC는 **canonical only**로 효과를 먼저 보고, augmentation은 게이트 통과 후 별도 설계 항목으로 미룬다. (configs의 `augmentation` 블록은 미사용 placeholder)

---

## File Structure

패키지 루트: `/path/to/lm_fusion`. 패키지명 `bpe_lm_fusion` (sibling project의 `text_only_bias` 컨벤션을 따름).

```
lm_fusion/
├── bpe_lm_fusion/
│   ├── __init__.py
│   ├── normalize.py        # 텍스트 정규화 규칙 (숫자/금액/공백)
│   ├── data.py             # HF dataset 로드, split별 text 추출
│   ├── corpus.py           # text -> WhisperTokenizer BPE pseudo-word corpus + augmentation
│   ├── kenlm_scorer.py     # KenLM 래퍼: id->word, state, log10->ln 점수
│   ├── fusion_processor.py # LogitsProcessor (top-k / full-vocab)
│   ├── decode.py           # Whisper generate + processor, n-best
│   ├── domain_terms.py     # 도메인 용어 lexicon 로드/매칭
│   ├── metrics.py          # CER/WER, term recall/precision, insertion
│   └── oracle.py           # n-best oracle term recall, first-subword rank CDF
├── scripts/
│   ├── build_corpus.py     # split text -> corpus 파일 (honest/ceiling)
│   ├── train_kenlm.sh      # lmplz + build_binary
│   ├── run_eval.py         # baseline vs fusion (alpha grid) 평가
│   └── run_rank_cdf.py     # first-subword rank CDF 분석
├── tests/
│   ├── test_normalize.py
│   ├── test_corpus.py
│   ├── test_kenlm_scorer.py
│   ├── test_fusion_processor.py
│   ├── test_metrics.py
│   └── test_oracle.py
├── configs/
│   └── poc.yaml            # 모델/alpha grid/경로/용어집 설정
└── data/                   # 생성물 (gitignore): corpus, *.arpa, *.binary, 결과 jsonl
```

각 파일 책임: **하나의 명확한 역할**. `corpus.py`는 토크나이즈+augmentation만, `kenlm_scorer.py`는 점수만, `fusion_processor.py`는 logits 결합만. 모델/데이터가 필요한 통합(`decode.py`, 스크립트)은 단위테스트 대신 스크립트 스모크로 검증.

---

## Track 1 — HuggingFace POC

### Task 0: 환경 설정

**Files:**
- Create: `bpe_lm_fusion/__init__.py` (빈 파일)
- Create: `configs/poc.yaml`
- Create: `.gitignore`

- [ ] **Step 1: 의존성 설치**

Run:
```bash
pip install "transformers>=4.44" torch datasets jiwer pyyaml numpy pytest
pip install https://github.com/kpu/kenlm/archive/master.zip   # python kenlm (scoring)
```

- [ ] **Step 2: KenLM toolkit 빌드 (lmplz / build_binary)**

Run:
```bash
cd /tmp && git clone https://github.com/kpu/kenlm.git
cd kenlm && mkdir -p build && cd build && cmake .. && make -j4
# lmplz, build_binary 가 /tmp/kenlm/build/bin 에 생성됨. PATH에 추가하거나 절대경로 사용.
echo 'export PATH=$PATH:/tmp/kenlm/build/bin' >> ~/.bashrc
```
Expected: `/tmp/kenlm/build/bin/lmplz` 실행 시 usage 출력.

- [ ] **Step 3: `configs/poc.yaml` 작성**

```yaml
model_name: openai/whisper-large-v3
language: korean
task: transcribe
dataset_path: /path/to/hf_dataset
train_split: train
test_split: validation     # validation을 test로 사용 (dataset_description.md §2)

kenlm:
  order: 5
  bin_dir: /tmp/kenlm/build/bin
  lmplz_extra: "--discount_fallback"   # 소규모 코퍼스 필수

fusion:
  alpha_grid: [0.0, 0.05, 0.1, 0.2, 0.4]
  asr_topk: 50
  mode: topk                  # topk | full_vocab

augmentation:
  bpe_dropout_p: 0.1
  samples_per_sentence: 2
  domain_term_samples: 4

paths:
  data_dir: ./data
domain_terms_file: ./configs/domain_terms.txt
max_audio_seconds: 30
```

- [ ] **Step 4: `.gitignore` 작성**

```gitignore
data/
__pycache__/
*.pyc
*.arpa
*.binary
.pytest_cache/
```

- [ ] **Step 5: `bpe_lm_fusion/__init__.py` 생성 (빈 파일) + 커밋**

```bash
touch bpe_lm_fusion/__init__.py
git add bpe_lm_fusion/__init__.py configs/poc.yaml .gitignore
git commit -m "chore: scaffold bpe_lm_fusion package + poc config"
```

---

### Task 1: 텍스트 정규화 (`normalize.py`)

**Files:**
- Create: `bpe_lm_fusion/normalize.py`
- Test: `tests/test_normalize.py`

baseline·fusion·정답·LM corpus에 **동일 적용**되는 정규화. POC 범위: 공백 정리 + 금액 콤마 제거 통일.

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_normalize.py
from bpe_lm_fusion.normalize import normalize_text

def test_collapse_whitespace():
    assert normalize_text("보장  개시일\t이후") == "보장 개시일 이후"

def test_strip_amount_commas():
    assert normalize_text("28,150원 입니다") == "28150원 입니다"

def test_strip_edges():
    assert normalize_text("  안녕하세요  ") == "안녕하세요"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL — `ModuleNotFoundError: bpe_lm_fusion.normalize`

- [ ] **Step 3: 구현**

```python
# bpe_lm_fusion/normalize.py
"""Shared text normalization. Apply identically to LM corpus, hyp, and ref."""
import re

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = _NUM_COMMA.sub("", text)   # 28,150 -> 28150
    text = _WS.sub(" ", text)         # collapse all whitespace to single space
    return text.strip()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add bpe_lm_fusion/normalize.py tests/test_normalize.py
git commit -m "feat: shared text normalization (whitespace + amount commas)"
```

---

### Task 2: 데이터 로드 (`data.py`)

**Files:**
- Create: `bpe_lm_fusion/data.py`
- Test: 없음 (실데이터 의존 → Task 7 스모크에서 검증)

- [ ] **Step 1: 구현**

```python
# bpe_lm_fusion/data.py
"""Load ASR dataset splits. See docs/dataset_description.md.

KenLM corpus는 train(+옵션 test) TEXT만 사용. 평가는 test audio+text.
audio는 raw float 리스트이므로 np.float32 변환 후 feature extractor에 투입.
"""
from __future__ import annotations
import numpy as np


def load_dataset(path: str):
    from datasets import load_from_disk
    return load_from_disk(path)


def split_texts(ds, split: str) -> list[str]:
    """Return list of raw transcript strings for a split."""
    return list(ds[split]["text"])


def audio_to_array(row) -> np.ndarray:
    return np.asarray(row["audio"], dtype=np.float32)
```

- [ ] **Step 2: import 스모크 + 커밋**

Run: `python -c "import bpe_lm_fusion.data; print('ok')"`
Expected: `ok`
```bash
git add bpe_lm_fusion/data.py && git commit -m "feat: dataset load + text/audio helpers"
```

---

### Task 3: BPE 코퍼스 생성 + augmentation (`corpus.py`)

**Files:**
- Create: `bpe_lm_fusion/corpus.py`
- Test: `tests/test_corpus.py`

text → `WhisperTokenizer` token id → `t<ID>` pseudo-word 라인. augmentation은 stochastic하되 **decode 검증 통과한 sample만** 채택.

- [ ] **Step 1: 실패 테스트 작성** (가짜 tokenizer로 로직만 검증 — 실모델 불필요)

```python
# tests/test_corpus.py
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# bpe_lm_fusion/corpus.py
"""Build BPE token-id pseudo-word corpus for KenLM.

각 transcript -> Whisper token id -> "t<ID> t<ID> ..." 한 줄.
augmentation: HuggingFace tokenizer는 BPE dropout 직접 노출이 없으므로,
POC에서는 sentencepiece 기반 대신 'canonical only'를 기본으로 하고,
augmentation 훅은 인터페이스만 제공(향후 dropout 토크나이저 연결 지점).
"""
from __future__ import annotations
from .normalize import normalize_text


def tokens_to_line(token_ids: list[int]) -> str:
    return " ".join(f"t{i}" for i in token_ids)


def text_to_canonical_line(text: str, tokenizer) -> str:
    text = normalize_text(text)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return tokens_to_line(ids)


def build_corpus(texts: list[str], tokenizer) -> list[str]:
    """Canonical corpus lines (one per text). Empty lines dropped."""
    lines = []
    for t in texts:
        line = text_to_canonical_line(t, tokenizer)
        if line:
            lines.append(line)
    return lines
```

> **범위 제외(추후 설계)**: 가이드 v2 §5~6의 stochastic BPE augmentation은 HF `WhisperTokenizer`(deterministic)로는 직접 안 된다. **POC는 canonical only로만 진행**하고, augmentation(별도 dropout 토크나이저 연결)은 **게이트 통과 후 별도 설계 항목**으로 미룬다. `build_corpus`는 인터페이스만 유지한다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_corpus.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add bpe_lm_fusion/corpus.py tests/test_corpus.py
git commit -m "feat: BPE token-id pseudo-word corpus builder"
```

---

### Task 4: 코퍼스 빌드 스크립트 (`scripts/build_corpus.py`)

**Files:**
- Create: `scripts/build_corpus.py`

honest(train) / ceiling(train+test) 두 코퍼스를 파일로 생성.

- [ ] **Step 1: 구현**

```python
# scripts/build_corpus.py
"""Write KenLM training corpora: honest (train) and ceiling (train+test).

Usage: python scripts/build_corpus.py --config configs/poc.yaml
"""
import argparse, os, yaml
from transformers import WhisperTokenizer
from bpe_lm_fusion.data import load_dataset, split_texts
from bpe_lm_fusion.corpus import build_corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    tok = WhisperTokenizer.from_pretrained(
        cfg["model_name"], language=cfg["language"], task=cfg["task"])
    ds = load_dataset(cfg["dataset_path"])
    train_txt = split_texts(ds, cfg["train_split"])
    test_txt = split_texts(ds, cfg["test_split"])

    out_dir = cfg["paths"]["data_dir"]
    os.makedirs(out_dir, exist_ok=True)

    honest = build_corpus(train_txt, tok)
    ceiling = build_corpus(train_txt + test_txt, tok)

    for name, lines in [("honest", honest), ("ceiling", ceiling)]:
        path = os.path.join(out_dir, f"corpus_{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{name}: {len(lines)} lines -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 + 산출물 확인**

Run: `python scripts/build_corpus.py --config configs/poc.yaml`
Expected: `honest: 2556 lines -> ./data/corpus_honest.txt` (대략), `ceiling: 2980 lines -> ...`

- [ ] **Step 3: 커밋**

```bash
git add scripts/build_corpus.py
git commit -m "feat: corpus build script (honest/ceiling)"
```

---

### Task 5: KenLM 학습 스크립트 (`scripts/train_kenlm.sh`)

**Files:**
- Create: `scripts/train_kenlm.sh`

- [ ] **Step 1: 구현**

```bash
#!/usr/bin/env bash
# Usage: scripts/train_kenlm.sh <corpus.txt> <out_prefix> <order> <bin_dir>
set -euo pipefail
CORPUS="$1"; OUT="$2"; ORDER="${3:-5}"; BIN="${4:-/tmp/kenlm/build/bin}"
# KenLM CLI는 시스템 libstdc++ 필요 (conda libstdc++ GLIBCXX_3.4.32 충돌 회피, Task 0 concern)
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"

"$BIN/lmplz" -o "$ORDER" --discount_fallback < "$CORPUS" > "${OUT}.arpa"
"$BIN/build_binary" "${OUT}.arpa" "${OUT}.binary"
echo "built ${OUT}.binary (order=$ORDER)"
```

- [ ] **Step 2: honest/ceiling LM 학습**

Run:
```bash
chmod +x scripts/train_kenlm.sh
scripts/train_kenlm.sh data/corpus_honest.txt  data/lm_honest_5g  5
scripts/train_kenlm.sh data/corpus_ceiling.txt data/lm_ceiling_5g 5
```
Expected: `built data/lm_honest_5g.binary (order=5)` 등. (`--discount_fallback` 덕에 소규모에서도 성공)

- [ ] **Step 3: 커밋**

```bash
git add scripts/train_kenlm.sh
git commit -m "feat: kenlm train script with discount_fallback"
```

---

### Task 6: KenLM 스코어러 (`kenlm_scorer.py`)

**Files:**
- Create: `bpe_lm_fusion/kenlm_scorer.py`
- Test: `tests/test_kenlm_scorer.py`

token id → `t<ID>` → KenLM 점수(log10→ln). history로 state 재계산(stateless).

- [ ] **Step 1: 실패 테스트 작성** (Task 5에서 만든 honest LM 사용)

```python
# tests/test_kenlm_scorer.py
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_kenlm_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError` (LM 있으면), 또는 skip (LM 없으면 → Task 5 먼저)

- [ ] **Step 3: 구현**

```python
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

    def start_state(self) -> "kenlm.State":
        s = kenlm.State()
        self.model.BeginSentenceWrite(s)   # python kenlm API (NOT BeginSentenceState)
        return s

    def advance(self, in_state, token_id: int):
        """Return (logprob_ln, out_state) after consuming token."""
        out = kenlm.State()
        lp10 = self.model.BaseScore(in_state, _word(token_id), out)
        return lp10 * LN10, out

    def state_from_history(self, token_ids):
        """Recompute state after a list of (non-special) token ids."""
        s = self.start_state()
        for t in token_ids:
            _, s = self.advance(s, t)
        return s

    def token_logprob(self, in_state, token_id: int) -> float:
        """ln-scale logprob of token given state (state not mutated)."""
        out = kenlm.State()
        return self.model.BaseScore(in_state, _word(token_id), out) * LN10

    # 테스트 편의용
    def most_common_first_token(self) -> int:
        return 0
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_kenlm_scorer.py -v`
Expected: PASS (2 passed) — Task 5 LM이 있을 때

- [ ] **Step 5: 커밋**

```bash
git add bpe_lm_fusion/kenlm_scorer.py tests/test_kenlm_scorer.py
git commit -m "feat: KenLM scorer (log10->ln, stateless history)"
```

---

### Task 7: Fusion LogitsProcessor (`fusion_processor.py`)

**Files:**
- Create: `bpe_lm_fusion/fusion_processor.py`
- Test: `tests/test_fusion_processor.py`

`log_softmax` 후 α·LM 가산. `mode=topk`면 ASR 상위 k에만, `full_vocab`이면 전체에. special token은 history에서 제외.

- [ ] **Step 1: 실패 테스트 작성** (가짜 스코어러로 로직 검증 — KenLM 불필요)

```python
# tests/test_fusion_processor.py
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_fusion_processor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_fusion_processor.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add bpe_lm_fusion/fusion_processor.py tests/test_fusion_processor.py
git commit -m "feat: BPE KenLM fusion LogitsProcessor (topk/full_vocab)"
```

---

### Task 8: 디코딩 파이프라인 (`decode.py`)

**Files:**
- Create: `bpe_lm_fusion/decode.py`
- Test: 없음 (모델 의존 → Task 10 스모크)

Whisper `generate` + processor. n-best 반환.

- [ ] **Step 1: 구현**

```python
# bpe_lm_fusion/decode.py
"""Whisper decoding with optional BPE-LM fusion. Returns 1-best + n-best."""
from __future__ import annotations
import torch
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          LogitsProcessorList)
from .fusion_processor import BpeKenlmFusionProcessor
from .data import audio_to_array


class FusionDecoder:
    def __init__(self, model_name: str, language: str, task: str, device: str = "cuda"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.language, self.task = language, task
        # skip set = special ids + timestamp ids (timestamps are NOT in all_special_ids)
        tok = self.processor.tokenizer
        skip = set(tok.all_special_ids)
        ts_begin = tok.convert_tokens_to_ids("<|0.00|>")
        if isinstance(ts_begin, int) and ts_begin > 0:
            skip |= set(range(ts_begin, self.model.config.vocab_size))
        self.skip_ids = skip

    def _features(self, row):
        wav = audio_to_array(row)
        feats = self.processor.feature_extractor(
            wav, sampling_rate=int(row["sampling_rate"]), return_tensors="pt"
        ).input_features
        return feats.to(self.device)

    @torch.no_grad()
    def transcribe(self, row, scorer=None, alpha=0.0, asr_topk=50,
                   mode="topk", num_beams=5, n_best=5):
        feats = self._features(row)
        procs = LogitsProcessorList()
        if scorer is not None and alpha > 0.0:
            procs.append(BpeKenlmFusionProcessor(
                scorer=scorer, alpha=alpha, asr_topk=asr_topk,
                skip_ids=self.skip_ids, mode=mode))
        gen = self.model.generate(
            feats, language=self.language, task=self.task,
            num_beams=num_beams, num_return_sequences=min(n_best, num_beams),
            logits_processor=procs, return_dict_in_generate=True)
        seqs = gen.sequences
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True)
        return {"best": texts[0], "nbest": texts}
```

- [ ] **Step 2: import 스모크 + 커밋**

Run: `python -c "import bpe_lm_fusion.decode; print('ok')"`
Expected: `ok`
```bash
git add bpe_lm_fusion/decode.py
git commit -m "feat: Whisper fusion decoder (1-best + n-best)"
```

---

### Task 9: 도메인 용어 + 지표 (`domain_terms.py`, `metrics.py`)

**Files:**
- Create: `bpe_lm_fusion/domain_terms.py`
- Create: `bpe_lm_fusion/metrics.py`
- Create: `configs/domain_terms.txt`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: 용어집 + 실패 테스트 작성**

```text
# configs/domain_terms.txt  (한 줄당 한 용어; test에서 추출 금지 — dataset_description §4B)
후유장해
보장개시일
급여상해수술비
간병보험
골절진단비
치주질환
```

```python
# tests/test_metrics.py
from bpe_lm_fusion.metrics import cer, term_recall_precision, insertion_rate

def test_cer_perfect():
    assert cer(["가나다"], ["가나다"]) == 0.0

def test_cer_one_sub():
    assert abs(cer(["가나다"], ["가라다"]) - 1/3) < 1e-9

def test_term_recall():
    refs = ["보장개시일 이후", "후유장해 발생"]
    hyps = ["보장개시일 이후", "후유 장해 발생"]   # 두번째는 띄어써서 미일치
    r = term_recall_precision(refs, hyps, ["보장개시일", "후유장해"])
    assert abs(r["recall"] - 0.5) < 1e-9

def test_insertion_rate_zero():
    assert insertion_rate(["가 나 다"], ["가 나 다"]) == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# bpe_lm_fusion/domain_terms.py
def load_terms(path: str) -> list[str]:
    terms = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms
```

```python
# bpe_lm_fusion/metrics.py
"""Eval metrics. CER/WER via jiwer; domain-term recall/precision; insertion."""
from __future__ import annotations
import jiwer


def cer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.cer(refs, hyps)


def wer(refs: list[str], hyps: list[str]) -> float:
    return jiwer.wer(refs, hyps)


def term_recall_precision(refs, hyps, terms) -> dict:
    """Occurrence-level: recall = matched/ref_total, precision = matched/hyp_total."""
    ref_total = hyp_total = matched = 0
    for ref, hyp in zip(refs, hyps):
        for term in terms:
            rc, hc = ref.count(term), hyp.count(term)
            ref_total += rc
            hyp_total += hc
            matched += min(rc, hc)
    recall = matched / ref_total if ref_total else 0.0
    precision = matched / hyp_total if hyp_total else 0.0
    return {"recall": recall, "precision": precision,
            "ref_total": ref_total, "hyp_total": hyp_total, "matched": matched}


def insertion_rate(refs, hyps) -> float:
    """jiwer 정렬 기반 insertion / ref word count."""
    out = jiwer.process_words(refs, hyps)
    n_ref = sum(len(r) for r in out.references)
    return out.insertions / n_ref if n_ref else 0.0
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add bpe_lm_fusion/domain_terms.py bpe_lm_fusion/metrics.py configs/domain_terms.txt tests/test_metrics.py
git commit -m "feat: domain terms + CER/WER/term-recall/insertion metrics"
```

---

### Task 10: oracle 지표 (`oracle.py`)

**Files:**
- Create: `bpe_lm_fusion/oracle.py`
- Test: `tests/test_oracle.py`

n-best oracle term recall (정답 용어가 n-best 중 하나라도 있으면 recoverable).

- [ ] **Step 1: 실패 테스트 작성**

```python
# tests/test_oracle.py
from bpe_lm_fusion.oracle import nbest_oracle_term_recall

def test_oracle_recovers_if_any_hyp_has_term():
    refs = ["보장개시일 안내"]
    nbests = [["보장 개시일 안내", "보장개시일 안내", "보장개일 안내"]]  # 2번째에 정답 용어
    r = nbest_oracle_term_recall(refs, nbests, ["보장개시일"])
    assert r["oracle_recall"] == 1.0

def test_oracle_miss():
    refs = ["보장개시일 안내"]
    nbests = [["보장 개시일", "보장개일"]]
    r = nbest_oracle_term_recall(refs, nbests, ["보장개시일"])
    assert r["oracle_recall"] == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
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
    return {"oracle_recall": recovered / ref_total if ref_total else 0.0,
            "ref_total": ref_total, "recovered": recovered}
```

- [ ] **Step 4: 테스트 통과 확인 + 커밋**

Run: `pytest tests/test_oracle.py -v`
Expected: PASS (2 passed)
```bash
git add bpe_lm_fusion/oracle.py tests/test_oracle.py
git commit -m "feat: n-best oracle term recall"
```

---

### Task 11: 평가 러너 (`scripts/run_eval.py`)

**Files:**
- Create: `scripts/run_eval.py`

baseline(α=0) vs fusion(α grid) × {honest, ceiling} 전체를 test에 돌려 지표 jsonl 출력.

- [ ] **Step 1: 구현**

```python
# scripts/run_eval.py
"""Run baseline vs fusion over alpha grid for honest & ceiling LMs.

Usage: python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest
출력: data/results_<cond>.jsonl  (alpha별 지표 한 줄)
"""
import argparse, json, os, yaml
from bpe_lm_fusion.data import load_dataset, audio_to_array  # noqa
from bpe_lm_fusion.decode import FusionDecoder
from bpe_lm_fusion.kenlm_scorer import KenlmScorer
from bpe_lm_fusion.domain_terms import load_terms
from bpe_lm_fusion.normalize import normalize_text
from bpe_lm_fusion import metrics, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--lm-cond", choices=["honest", "ceiling"], default="honest")
    ap.add_argument("--limit", type=int, default=0, help="0=full test set")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    ds = load_dataset(cfg["dataset_path"])
    test = ds[cfg["test_split"]]
    if args.limit:
        test = test.select(range(args.limit))
    terms = load_terms(cfg["domain_terms_file"])

    dec = FusionDecoder(cfg["model_name"], cfg["language"], cfg["task"])
    lm_path = os.path.join(cfg["paths"]["data_dir"], f"lm_{args.lm_cond}_5g.binary")
    scorer = KenlmScorer(lm_path)

    refs = [normalize_text(r["text"]) for r in test]
    out_path = os.path.join(cfg["paths"]["data_dir"], f"results_{args.lm_cond}.jsonl")
    fout = open(out_path, "w")

    for alpha in cfg["fusion"]["alpha_grid"]:
        hyps, nbests = [], []
        for row in test:
            res = dec.transcribe(
                row, scorer=scorer, alpha=alpha,
                asr_topk=cfg["fusion"]["asr_topk"], mode=cfg["fusion"]["mode"],
                num_beams=5, n_best=5)
            hyps.append(normalize_text(res["best"]))
            nbests.append([normalize_text(t) for t in res["nbest"]])
        rec = {
            "lm_cond": args.lm_cond, "alpha": alpha,
            "cer": metrics.cer(refs, hyps), "wer": metrics.wer(refs, hyps),
            "insertion_rate": metrics.insertion_rate(refs, hyps),
            **{f"term_{k}": v for k, v in
               metrics.term_recall_precision(refs, hyps, terms).items()},
            **{f"oracle_{k}": v for k, v in
               oracle.nbest_oracle_term_recall(refs, nbests, terms).items()},
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print(rec)
    fout.close()
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 작은 스모크 실행 (limit=5)**

Run: `python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest --limit 5`
Expected: alpha별 dict 5줄 출력, `data/results_honest.jsonl` 생성. 에러 없이 완주.

- [ ] **Step 3: 커밋**

```bash
git add scripts/run_eval.py
git commit -m "feat: eval runner (alpha grid, honest/ceiling)"
```

---

### Task 12: 실험 실행 + 판정

**Files:**
- 산출물: `data/results_honest.jsonl`, `data/results_ceiling.jsonl`

- [ ] **Step 1: 전체 평가 실행 (honest + ceiling)**

Run:
```bash
python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest
python scripts/run_eval.py --config configs/poc.yaml --lm-cond ceiling
```

- [ ] **Step 2: 효과 게이트 판정** (가이드 v2 §22)

다음을 honest 기준으로 확인:
```text
domain term recall  : +3~5%p 이상 (alpha 0.05~0.2 구간)
CER                 : 악화 없음 또는 미미 (+0.002 이하)
insertion_rate      : 급증 없음
ceiling vs honest   : ceiling이 honest보다 크게 높으면 -> "외부 도메인 텍스트 확보 시 상승 여력" 신호
```

- [ ] **Step 3: rank CDF 분석 (선택, union/research 진행 판단)**

Run: `python scripts/run_rank_cdf.py --config configs/poc.yaml --limit 100` (Task 13)
판정: 정답 첫 subword가 top-16 안 → top-k-only 가능 / top-64 밖 다수 → 외부 LM보다 acoustic adaptation 우선.

- [ ] **Step 4: 결과 요약 문서화**

`docs/poc_results.md`에 alpha 곡선·honest/ceiling 비교·게이트 통과 여부 기록.

---

### Task 13: first-subword rank CDF (`scripts/run_rank_cdf.py`)

**Files:**
- Create: `scripts/run_rank_cdf.py`

teacher-forcing으로 reference 토큰열을 넣고, 각 도메인 용어 첫 subword의 ASR logits 내 rank 분포 수집.

- [ ] **Step 1: 구현**

```python
# scripts/run_rank_cdf.py
"""정답 도메인 용어의 '첫 subword'가 ASR logits에서 몇 위였는지 분포(CDF).

teacher forcing: decoder_input_ids = reference, 각 position logits에서
다음 정답 토큰의 rank를 구한다. 용어 첫 subword position만 모은다.
"""
import argparse, json, os, yaml, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from bpe_lm_fusion.data import load_dataset, audio_to_array
from bpe_lm_fusion.domain_terms import load_terms
from bpe_lm_fusion.normalize import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/poc.yaml")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    proc = WhisperProcessor.from_pretrained(cfg["model_name"])
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float16).to("cuda").eval()  # fp16: GPU VRAM
    tok = proc.tokenizer
    tok.set_prefix_tokens(language=cfg["language"], task=cfg["task"], predict_timestamps=False)

    ds = load_dataset(cfg["dataset_path"])
    test = ds[cfg["test_split"]].select(range(min(args.limit, len(ds[cfg["test_split"]]))))
    terms = load_terms(cfg["domain_terms_file"])
    # 용어를 in-context 토큰열로 준비. Whisper byte-level BPE는 단어 시작에 공백마커가 붙어,
    # 단독 토큰화와 문장 중간(앞 공백) 토큰화의 첫 subword id가 다르다.
    # -> 앞공백 포함/미포함 두 변형을 모두 후보로 두고, label 토큰열에서 '부분열'로 매칭한다.
    term_token_seqs = []
    for t in terms:
        tnorm = normalize_text(t)
        term_token_seqs.append({
            tuple(tok(" " + tnorm, add_special_tokens=False).input_ids),  # mid-sentence
            tuple(tok(tnorm, add_special_tokens=False).input_ids),         # sentence-initial
        })

    ranks = []
    for row in test:
        ref = normalize_text(row["text"])
        if not any(t in ref for t in terms):
            continue
        feats = proc.feature_extractor(audio_to_array(row),
                sampling_rate=int(row["sampling_rate"]), return_tensors="pt"
                ).input_features.to("cuda", dtype=torch.float16)
        labels = tok(text_target=ref, return_tensors="pt").input_ids.to("cuda")  # 특수 prefix 포함
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=labels[:, :-1]).logits[0]
        tgt = [int(x) for x in labels[0, 1:].tolist()]
        # 각 용어 토큰열이 tgt의 부분열로 나타나는 위치에서, 첫 subword의 rank를 기록
        for seqs in term_token_seqs:
            for seq in seqs:
                L = len(seq)
                for i in range(len(tgt) - L + 1):
                    if tuple(tgt[i:i + L]) == seq:
                        first_id = seq[0]
                        rank = int((logits[i] > logits[i, first_id]).sum()) + 1  # 1-based
                        ranks.append(rank)
    ranks.sort()
    buckets = {"<=8": 0, "9-16": 0, "17-64": 0, ">64": 0}
    for r in ranks:
        k = "<=8" if r <= 8 else "9-16" if r <= 16 else "17-64" if r <= 64 else ">64"
        buckets[k] += 1
    out = {"n": len(ranks), "buckets": buckets}
    os.makedirs(cfg["paths"]["data_dir"], exist_ok=True)
    json.dump(out, open(os.path.join(cfg["paths"]["data_dir"], "rank_cdf.json"), "w"),
              ensure_ascii=False, indent=2)
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 + 커밋**

Run: `python scripts/run_rank_cdf.py --config configs/poc.yaml --limit 100`
Expected: `{"n": ..., "buckets": {"<=8": ..., ...}}`
```bash
git add scripts/run_rank_cdf.py
git commit -m "feat: first-subword rank CDF analysis"
```

---

## Track 1 게이트 → Track 2 진입 조건

**다음을 모두 만족할 때만 Track 2(CT2) 착수:**
- honest 조건에서 domain term recall **+3%p 이상**, CER 악화 미미, insertion 급증 없음
- (또는) ceiling이 뚜렷이 높아 **외부 텍스트 확보 시 상승 여력**이 확인됨
- rank CDF에서 정답 첫 subword가 **top-k 내(주로 ≤64)** → 1-pass fusion이 의미 있는 영역

게이트 미달이면: 외부 도메인 텍스트 확충 후 재측정 / 또는 word-level 2-pass rerank(분석 문서의 MVP)로 선회 / 또는 acoustic-side adaptation 검토.

---

## Track 2 — CT2 Serving (게이트 통과 시 **별도 상세 플랜**)

> 이 트랙은 독립 서브시스템(C++ 포크)이며, HF 게이트 통과 후 `docs/implementation_plan_ct2.md`로 별도 작성한다. 여기서는 범위만 고정한다.

**범위 (가이드 v2 §13~16):**
1. CT2 C++ beam search에 **per-beam KenLM state** + fusion 삽입 (`FullScore`, WordIndex 매핑 테이블, `BeginSentenceState`).
2. **top-k only → union(domain successor)** 순으로 단계 확장.
3. **GPU/CPU 경계**: GPU에서 ASR top-k만 추출 → CPU에서 KenLM 점수 → beam 선택. beam reorder 시 LM state 동기 reorder.
4. **special/timestamp token** LM 점수 0 처리.
5. **length penalty / score normalization ablation** (가이드 v2 §11).
6. **latency p95 / throughput** 측정, **feature flag / fallback**, hardening.

**Track 2 산출물:** CT2 fork 빌드, C++ 단위테스트(LM state reorder), latency 벤치, 운영 가이드.

---

## Self-Review (작성자 점검 결과)

- **Spec coverage**: 코퍼스 생성(Task 3~4)·KenLM 학습(Task 5)·스코어러(Task 6)·fusion processor(Task 7)·디코딩(Task 8)·지표(Task 9~10)·실험/게이트(Task 11~13) → 가이드 v2의 §7,9,10,12,14,20,22 및 분석 문서의 oracle/rank-CDF 권고를 모두 커버. CT2(가이드 §13~16)는 Track 2로 분리.
- **누수 2조건(honest/ceiling)**: Task 4에서 코퍼스 분리, Task 11에서 `--lm-cond`로 양쪽 실행 → 합의사항 반영.
- **Placeholder scan**: 모든 코드 step에 실제 코드 포함. "적절히 처리" 류 없음.
- **Type consistency**: `KenlmScorer.state_from_history`/`token_logprob`가 Task 6 정의 ↔ Task 7 processor 사용 시그니처 일치. `term_recall_precision` 반환 키(`recall`/`precision`)가 Task 9 정의 ↔ Task 11 사용 일치.
- **알려진 제약(의도적)**: stochastic BPE augmentation은 POC 범위 제외(추후 설계), Task 3은 canonical only.
- **성능 주의(리뷰 반영)**: POC 스코어러는 매 step `state_from_history`로 history 전체를 재계산(O(L²)/seq). `topk` 모드는 full 424 test에서 수 분~수십 분 수준으로 완주 가능하나, `full_vocab` 모드(≈51865 × beam KenLM 호출/step)는 사실상 실행 불가 → **`full_vocab`은 `--limit` 소규모 진단 전용**. CT2(Track 2)에서 per-beam state 증분 갱신으로 최적화.
- **외부 리뷰 반영**: BLOCKER 2건(`BeginSentenceWrite`, scores에 직접 가산 + 마스크 보존), MAJOR(super().__init__, timestamp skip, Task 13 앞공백 부분열 매칭) 수정 완료. 검증된 정상 항목: `generate` kwargs / beam shape / `num_return_sequences=min` / 배치 input_features / jiwer 속성.
