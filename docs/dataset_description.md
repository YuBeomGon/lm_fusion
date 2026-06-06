# 데이터셋 설명 (LM Fusion 관점 — 보험 통화 STT)

이 문서는 BPE token-level KenLM **1-pass fusion**([bpe_kenlm_1pass_fusion_guide_v2.md](./bpe_kenlm_1pass_fusion_guide_v2.md)) 및 word/BPE LM 분석([assets/Fusion Analysis ...](./assets/Fusion%20Analysis%20of%20BPE%20Token%20LM%20and%20Word%20LM%20for%20Whisper%20Domain%20Adaptation%20on%20CT2.md))에서 사용할 데이터셋을 LM fusion 관점에서 정리한 것이다.

- **데이터셋 정본**: `sibling project`과 **동일한 보험 통화 STT 데이터셋**을 공유한다.
- **원본 설명서(정본)**: [`<SIBLING_PROJECT>/docs/dataset_description.md`](<SIBLING_PROJECT>/docs/dataset_description.md) — 규모/스키마/분포/leakage 상세는 그 문서를 정본으로 따른다. 본 문서는 **LM fusion이 그 데이터를 어떻게 쓰는지**만 다룬다.
- **확인일**: 2026-06-06

---

## 0. text-only bias와 무엇이 다른가

| | text-only bias (sibling project) | **LM fusion (이 프로젝트)** |
|---|---|---|
| train text 용도 | bias 모듈 `B_H` 학습 | **KenLM(n-gram LM) 학습** |
| 모델 변경 | decoder 내부 (cross-attn bias) | **모델 외부 LM을 디코딩 점수에 결합** |
| 평가 데이터 | test audio+text | 동일 (test audio+text) |
| 추가로 필요한 것 | — | ① BPE token-id corpus ② 도메인 용어 lexicon ③ oracle 분석용 n-best |

→ **데이터 출처·split·leakage 규칙은 sibling project과 동일**하게 따른다. 다른 점은 "train text를 무엇으로 변환해 쓰느냐"뿐이다.

---

## 1. 출처 / 로드

- **경로**: `/path/to/hf_dataset`
- **형식**: HuggingFace `datasets` `DatasetDict` (Arrow, `load_from_disk`)
- **도메인**: 한국어 **보험 텔레마케팅 통화** 전사

```python
from datasets import load_from_disk
ds = load_from_disk("/path/to/hf_dataset")
# DatasetDict({ train, validation })
```

핵심 컬럼만:
- **`text`** (string) — 정답 transcript. **KenLM 학습/평가의 핵심.**
- **`audio`** (`Sequence[float32]`) — raw waveform 리스트 (HF `Audio` feature 아님). 평가 시 `np.array(row["audio"], dtype=np.float32)` 변환 후 feature extractor 투입.
- **`sampling_rate`** = 16000 고정.
- 나머지(`source_wav`, `chunk_id`, `cut_*` 등)는 출처/정렬 메타데이터. 상세는 정본 문서 §3.

---

## 2. Split 운용 방침 (sibling project과 동일)

> `train`(2,556) → **train** (KenLM 학습, **text만** 사용)
> `validation`(424) → **test** (baseline vs fusion 비교)
> **valid split 없음.**

- 통화(`source_wav`) 단위로 train↔test가 분리되어 있어 leakage 0건 (정본 §5). validation을 test로 그대로 써도 안전.
- **`alpha` 선택**: valid가 없으므로 best 1개를 사후 선택하지 않고, test에서 **alpha grid 전체 곡선을 리포트**한다 (정본 §6 방식 A 계승). `alpha=0.0`(fusion off)을 baseline sanity check로 반드시 포함.

---

## 3. ⚠️ 가장 중요한 제약 — 코퍼스 규모

가이드(v2 §18) 기준 텍스트량 등급:

| 등급 | 문장 수 | train | 판정 |
|---|---|---|---|
| smoke test | 1k~5k | **2,556** | ← **여기** |
| 유의미한 domain LM | 10k~50k | — | 미달 |
| 안정적 5-gram | 50k~100k+ | — | 미달 |

- train text는 **2,556 문장(중앙값 ~124자)** → BPE 토큰으로 펼쳐도 **5-gram을 안정적으로 추정하기엔 sparse**하다.
- 결론: **train text만으로는 smoke test(파이프라인 검증)까지만** 가능. 유의미한 도메인 LM을 만들려면 **외부 도메인 텍스트 추가가 필수**.

### 추가 텍스트 소스 (수집 대상 — 결정 필요)
가이드 §19 / 분석 문서 "실험 설계"의 권장 비율(실제 transcript·스크립트 70~90% + synthetic 10~30%)을 따른다.
- 보험 **약관 / 상품설명서** 문장
- 상담 **스크립트 / FAQ**
- 도메인 용어 포함 **synthetic template** (과반복 시 boilerplate 과발화 주의)

> 📌 **열린 결정사항**: 외부 텍스트를 어디서 얼마나 확보할지. 1k → 10k → 100k 구간으로 늘리며 n-gram order(3/5)와 함께 sweep (분석 문서 실험 grid).

---

## 4. LM fusion이 데이터를 쓰는 경로

### (A) KenLM 학습 corpus — **train text only**
```
train text
  → 정규화 (§5)
  → Whisper tokenizer로 BPE token id 변환  (decode 시점과 동일 tokenizer 필수)
  → "t<ID> t<ID> ..." pseudo-word corpus  (가이드 v2 §7)
  → (옵션) stochastic BPE augmentation     (가이드 v2 §5~6, A/B로 검증)
  → lmplz -o 5 --discount_fallback         (소규모 코퍼스 필수 플래그)
```
- **POC(HuggingFace)**: corpus 생성·디코딩 모두 `WhisperTokenizer`로 통일.
- special token(SOT/language/task/timestamp)은 corpus에서 제외 (가이드 v2 §7).

### (B) 도메인 용어 lexicon — domain-term recall 평가용
- 출처: **train text + 보험 도메인 용어집** (예: 급여상해수술비, 간병보험, 골절 진단비, 보장개시일, 후유장해, 치주질환 …)
- **test transcript에서 용어를 추출하면 안 됨** (leakage). 평가 항목 정의에만 test를 쓴다.

### (C) 평가 — test audio + text
- baseline(fusion off) vs fusion(alpha grid) 비교.
- **oracle 분석**: test에서 n-best 추출 → reference 용어가 n-best 안에 있는지(**n-best oracle term recall**), 첫 정답 subword의 **rank CDF** (분석 문서 권고: 이 두 지표로 top-k-only/union/research 진행 여부 결정).

---

## 5. 텍스트 정규화 (baseline·fusion·정답 동일 적용)

전사에 숫자/금액/단위가 빈번(`28,150원`, `50,000`, `%`). 다음을 **모든 경로에 동일하게** 적용:
- 숫자·금액 표기 통일, 단위 표기 통일
- 영문 대문자/브랜드코드 정규화
- 공백(띄어쓰기) 변이 통일

> word LM 분석편에서 지적했듯, 한국어는 eojeol 경계 granularity가 다양해 **정규화 품질이 LM 품질을 좌우**한다. raw eojeol 버전과 정규화 버전을 모두 준비해 비교 권장(분석 문서 실험 설계).

---

## 6. 데이터 처리 주의

1. **30초 초과 chunk (test 2건, 최대 45.08s)** — Whisper 30s window 초과. 30s로 분할하거나 제외 (정본 §6).
2. **audio 변환** — raw float 리스트 → `np.array(..., dtype=np.float32)`.
3. **train 내부 overlap** — train은 `duration_aware_overlap`이라 chunk 간 시간 겹침 가능. KenLM 학습엔 무방하나, 같은 문장 과중복은 LM 빈도 왜곡 → augmentation/oversampling 시 주의.
4. **민감정보** — 실제 통화 전사는 고객 민감정보. 공개 저장소 반출 금지, 로컬에서만 사용.

---

## 7. Leakage 체크리스트 (LM fusion)

- [ ] KenLM 학습 corpus에 **test(validation) transcript 미포함**
- [ ] 도메인 용어 lexicon을 **test에서 추출하지 않음**
- [ ] 외부 텍스트 추가 시, test 통화와 동일 출처/중복 문장 없는지 확인
- [ ] tokenizer가 학습 corpus 생성 시점과 디코딩 시점에 **동일**

---

## 8. 열린 결정사항 (다음 단계)

1. **외부 도메인 텍스트 확보** — 규모(목표 10k~50k+)와 소스(약관/스크립트/FAQ/synthetic).
2. **정규화 규칙 확정** — 숫자/금액/공백 표준.
3. **도메인 용어 lexicon 확정** — 평가 대상 critical term 목록.
4. **tokenizer 고정** — POC는 HF `WhisperTokenizer`, CT2 이식 시 id 매핑 동일성 확인.
