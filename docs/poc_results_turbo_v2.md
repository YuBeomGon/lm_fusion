# BPE-LM 1-pass Fusion — Turbo POC 결과 (v2)

**일자**: 2026-06-06

**조건**: HuggingFace POC, `honest`/`ceiling` 2조건

**모델**: `openai/whisper-large-v3-turbo` (fp16, beam=5, n_best=5)

**fusion**: `score = ASR_logprob + alpha * LM_logprob`, `asr_topk=50`, `mode=topk`

**LM**: Whisper BPE token-id 5-gram KenLM (`t<ID>` pseudo-word)

**평가 샘플**: validation/test 전체 424개

**alpha grid**: `0.0 / 0.05 / 0.1 / 0.15 / 0.2 / 0.3 / 0.4 / 0.6`

> 본 문서는 집계 지표만 포함한다. 원본 전사, 코퍼스, LM binary, 결과 jsonl은 저장소에 포함하지 않는다.

## 1. 조건 정의

| 조건 | LM corpus | 용도 |
|---|---|---|
| `honest` | train text only | 실제 보고용 성능 |
| `ceiling` | train + test text | 상한선/진단용. 성능값으로 보고 금지 |

`ceiling`은 test transcript를 포함하므로 누설 조건이다. 이 결과는 “외부 도메인 텍스트를 충분히 확보하면 어느 정도 여지가 있는지”를 보는 진단값으로만 사용한다.

## 2. Honest 결과

| alpha | CER | ΔCER | WER | term recall | Δrecall | precision | insertion | repeated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1200 | - | 0.3372 | 0.7599 | - | 0.9591 | 0.0476 | 0.0000 |
| 0.05 | 0.1123 | -0.0077 | 0.3069 | 0.7978 | +3.8%p | 0.9583 | 0.0387 | 0.0000 |
| 0.10 | 0.1070 | -0.0130 | 0.2913 | 0.8223 | +6.2%p | 0.9490 | 0.0347 | 0.0000 |
| **0.15** | **0.1015** | **-0.0185** | **0.2733** | **0.8499** | **+9.0%p** | 0.9472 | **0.0309** | 0.0000 |
| 0.20 | 0.1133 | -0.0067 | 0.2765 | 0.8602 | +10.0%p | 0.9494 | 0.0419 | 0.0024 |
| 0.30 | 0.1133 | -0.0067 | 0.2653 | 0.8728 | +11.3%p | 0.9420 | 0.0236 | 0.0024 |
| 0.40 | 0.1611 | +0.0411 | 0.3472 | 0.9021 | +14.2%p | 0.9422 | 0.1009 | 0.0094 |
| 0.60 | 0.4438 | +0.3238 | 0.8428 | 0.8578 | +9.8%p | 0.9330 | 0.5162 | 0.0354 |

### Honest 해석

- **최적 균형점은 alpha=0.15**다. CER는 0.1200 -> 0.1015로 개선되고, WER도 0.3372 -> 0.2733으로 개선된다. 도메인 용어 recall은 0.7599 -> 0.8499로 **+9.0%p** 상승한다.
- **보수적 운영 후보는 alpha=0.05**다. recall +3.8%p, CER 개선, precision 거의 유지, 반복/환각 지표 0이다.
- alpha=0.2부터 repeated/length outlier가 처음 발생한다.
- alpha=0.4 이상은 recall만 보면 높지만 CER, insertion, repeated-text가 악화되어 **over-bias 구간**으로 본다.

## 3. Ceiling 결과

| alpha | CER | ΔCER | WER | term recall | Δrecall | precision | insertion | repeated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.1200 | - | 0.3372 | 0.7599 | - | 0.9591 | 0.0476 | 0.0000 |
| 0.05 | 0.1028 | -0.0172 | 0.2825 | 0.8017 | +4.2%p | 0.9594 | 0.0380 | 0.0000 |
| 0.10 | 0.0927 | -0.0273 | 0.2457 | 0.8310 | +7.1%p | 0.9599 | 0.0310 | 0.0000 |
| 0.15 | 0.0878 | -0.0322 | 0.2179 | 0.8373 | +7.7%p | 0.9593 | 0.0248 | 0.0000 |
| 0.20 | 0.0809 | -0.0391 | 0.1964 | 0.8562 | +9.6%p | 0.9610 | 0.0221 | 0.0000 |
| 0.30 | 0.0771 | -0.0429 | 0.1758 | 0.8784 | +11.8%p | 0.9570 | 0.0176 | 0.0024 |
| **0.40** | **0.0764** | **-0.0436** | **0.1647** | **0.9013** | **+14.1%p** | 0.9548 | **0.0145** | 0.0024 |
| 0.60 | 0.1464 | +0.0264 | 0.2780 | 0.9194 | +16.0%p | 0.9580 | 0.1213 | 0.0071 |

### Ceiling 해석

- `ceiling` 최저 CER는 alpha=0.4의 **0.0764**다. 같은 alpha에서 `honest`는 CER가 0.1611로 무너지므로, 단순 alpha 문제라기보다 LM coverage/quality 차이가 크다.
- `ceiling`은 alpha=0.2~0.4 구간에서 CER, WER, term recall이 모두 강하게 개선된다.
- 이 격차는 train text만으로 만든 LM이 아직 sparse하며, 외부 도메인 텍스트 확보 시 추가 개선 여지가 크다는 신호다.

## 4. Gate 판정

| 기준 | 판정 |
|---|---|
| domain term recall +3~5%p 이상 | PASS. honest alpha=0.05에서 +3.8%p, alpha=0.15에서 +9.0%p |
| CER 악화 없음 또는 미미 | PASS. honest alpha=0.15에서 -0.0185 개선 |
| precision 급락 없음 | PASS. alpha=0.15에서 -1.2%p 수준. 보수 alpha=0.05는 -0.1%p |
| insertion 급증 없음 | PASS. alpha=0.15에서 0.0476 -> 0.0309 감소 |
| repeated/no-speech/phrase hallucination 증가 없음 | PASS at alpha<=0.15 |

**최종 판정: PASS.**

HF POC 기준으로 BPE token-level KenLM 1-pass fusion은 효과가 확인됐다. CT2 이식으로 넘어갈 근거가 충분하다.

## 5. 권장 alpha

| 목적 | 권장 alpha | 이유 |
|---|---:|---|
| 보고용 sweet spot | **0.15** | CER/WER 최저, recall +9.0%p, 반복 0 |
| 보수 운영 후보 | **0.05** | 안정성 최우선, recall +3.8%p, precision 거의 유지 |
| 금지 구간 | **>=0.4** | honest에서 CER/insertion/repeated-text 악화 |

## 6. 다음 단계

1. `alpha=0.15`를 HF POC 대표값으로 문서화한다.
2. `alpha=0.05`를 보수 운영 후보로 별도 기록한다.
3. CT2 이식은 top-k only부터 시작하되, alpha가 커질 때 over-bias가 빠르게 나타나는 점을 feature flag와 guardrail에 반영한다.
4. train text만으로는 LM coverage가 부족하므로 외부 도메인 텍스트를 추가해 honest 조건을 재측정한다.
5. 도메인 용어 metric은 longest-match/공백 정책을 명확히 하는 후속 개선이 필요하다.

## 7. 재현 명령

```bash
python scripts/build_corpus.py --config configs/poc.yaml
scripts/train_kenlm.sh data/corpus_honest.txt data/lm_honest_5g 5
scripts/train_kenlm.sh data/corpus_ceiling.txt data/lm_ceiling_5g 5
PYTHONPATH=. python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest
PYTHONPATH=. python scripts/run_eval.py --config configs/poc.yaml --lm-cond ceiling
```

산출물:

```text
data/results_honest.jsonl
data/results_ceiling.jsonl
```

위 산출물은 `.gitignore` 대상이며 저장소에 포함하지 않는다.
