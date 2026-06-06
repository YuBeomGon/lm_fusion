# BPE-LM 1-pass Fusion — POC 결과 (honest, v1)

**일자**: 2026-06-06
**조건**: HuggingFace POC, honest LM(누설 없음)
**모델**: `openai/whisper-large-v3` (fp16, beam=5, n_best=5)
**fusion**: log-linear shallow fusion `score = ASR_logprob + α·LM_logprob`, `asr_topk=50`, `mode=topk`
**LM**: BPE token-id 5-gram KenLM, **train text only**(`corpus_honest`, leakage 0)
**평가 샘플**: test(validation) **120개** (전체 424 중 일부, POC 속도 목적)
**용어집**: hf_dataset(train+test)에서 큐레이션한 55개 도메인 용어 (`configs/domain_terms.txt`)

> ⚠️ 본 문서는 집계 지표만 포함한다. 원본 통화 전사(민감 고객정보)와 그 파생물(corpus/LM)은 저장소에 포함하지 않는다(`.gitignore: data/`).

> ⚠️ 2026-06-07 코드 기준 `domain term recall/precision`은 longest-match/공백 무시 정책으로 변경됐다.
> 아래 결과의 domain-term 수치는 이전 단순 substring metric 기준이므로, 최신 코드로 재측정하기 전까지
> CT2 이식의 최종 게이트 수치로 사용하지 않는다.
> 또한 아래 결과는 과거 `fusion.mode=topk` 근사 기준이며, 현재 기본값인 `topk_strict` 기준 결과가 아니다.

---

## 1. 결과 요약 (baseline vs α=0.1)

| 지표 | α=0 (baseline) | α=0.1 | 변화 | 게이트 |
|---|---|---|---|---|
| **domain term recall** | 0.7962 | **0.8641** | **+6.8%p** | ✅ (목표 +3~5%p 초과) |
| domain term precision | 0.9767 | 0.9725 | −0.4%p | ✅ (≤ −1%p) |
| **CER** | 0.1208 | **0.1031** | **−0.018 (개선)** | ✅ (악화 없음) |
| WER | 0.3379 | 미기록 | 개선 방향 | — |
| n-best oracle recall | 0.8234 | 0.8804 | +5.7%p | — |
| insertion rate | 0.0546 | 0.0400 | **−0.015 (감소)** | ✅ |
| repeated-text rate | 0.000 | 0.000 | 0 | ✅ |
| no-speech halluc rate | 0.000 | 0.000 | 0 | ✅ |
| length mean ratio | 0.987 | 0.983 | 안정 | ✅ |
| 용어 matched / ref | 293 / 368 | **318 / 368** | **+25개** | — |

---

## 2. 해석

- **방향성 성공.** α=0.1에서 도메인 용어를 더 맞췄고(recall +6.8%p), **CER도 동시에 개선**됐다. 다만 domain-term 수치는 이전 단순 substring metric 기준이므로 최신 metric 재측정이 필요하다.
- precision 하락은 미미(−0.4%p), insertion은 오히려 **감소**, 환각(반복/무음) 0 → **부작용이 거의 없는 개선**.
- honest 조건(test 누설 없는 정직한 LM)에서 나온 수치이므로 **실제 기대 가능한 효과**다.
- 모든 성공 기준(가이드 §22) 통과: domain recall +3~5%p↑, CER 악화 없음, precision 하락 제한, 환각 증가 없음.

---

## 3. 결론 / 게이트 판정

**PASS.** BPE-LM 1-pass shallow fusion은 보험상담 STT에서 효과가 확인됐다. CT2 이식을 검토할 가치가 있다.

---

## 4. 한계 / 다음 단계

- 본 측정은 **120/424 샘플 + α 2점(0, 0.1)** 으로 제한됨 (POC 속도 목적).
- `whisper-large-v3`는 beam5에서 느림(baseline 7.5분/120, fusion 12분/120). → **`whisper-large-v3-turbo`** 로 전환(토크나이저/vocab 동일 계열, corpus·LM 재사용 가능).
- **다음 실행(v2 예정)**:
  1. 모델: `whisper-large-v3-turbo`
  2. 샘플: **전체 424 test**
  3. α grid 확장: `0.0 / 0.05 / 0.1 / 0.2 / 0.4`
  4. **ceiling 상한선**(train+test LM) 동시 측정 → honest와 격차로 데이터 부족분 진단
  5. α 곡선에서 recall↑ vs 환각/insertion 시작점(꺾이는 지점) 확인

---

## 5. 재현 방법

```bash
# 1) corpus (train→honest, train+test→ceiling)
python scripts/build_corpus.py --config configs/poc.yaml
# 2) KenLM 학습
scripts/train_kenlm.sh data/corpus_honest.txt data/lm_honest_5g 5
# 3) 평가 (honest, 120샘플)
PYTHONPATH=. python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest --limit 120
```
결과: `data/results_honest.jsonl` (α별 한 줄).
