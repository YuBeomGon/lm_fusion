# lm_fusion — BPE token-LM 1-pass shallow fusion for Whisper

Whisper ASR 디코딩에 **BPE token-id n-gram KenLM**을 1-pass shallow fusion으로 결합해
도메인 long-tail 용어 인식을 개선하는 실험 코드 (HuggingFace POC).

```
score(token) = ASR_logprob + α · LM_logprob
```

LM은 Whisper tokenizer의 BPE token id를 의사단어(`t<ID>`)로 펼친 코퍼스로 학습하므로,
**첫 subword부터** 도메인 prior가 beam search에 개입한다 (word-level LM 대비 장점).

## 구성

| 경로 | 역할 |
|---|---|
| `bpe_lm_fusion/corpus.py` | transcript → BPE token-id 코퍼스 (`t<ID> ...` 한 줄/문장) |
| `bpe_lm_fusion/kenlm_scorer.py` | KenLM 상태 기반 token logprob (log10→ln 변환) |
| `bpe_lm_fusion/fusion_processor.py` | HF `LogitsProcessor` — `topk_strict`/`topk`/`full_vocab` fusion |
| `bpe_lm_fusion/decode.py` | Whisper(fp16) 디코더 + fusion, 1-best/n-best |
| `bpe_lm_fusion/metrics.py` | CER/WER, domain-term recall·precision, insertion, 환각/안전 지표 |
| `bpe_lm_fusion/oracle.py` | n-best oracle term recall |
| `scripts/` | corpus 빌드 / KenLM 학습 / 평가 / rank-CDF |
| `docs/` | 설계 가이드, 분석, 구현 플랜, POC 결과 |

## 빠른 실행

```bash
python scripts/build_corpus.py --config configs/poc.yaml
scripts/train_kenlm.sh data/corpus_honest.txt data/lm_honest_5g 5
PYTHONPATH=. python scripts/run_eval.py --config configs/poc.yaml --lm-cond honest --limit 120
```

설정은 `configs/poc.yaml`에서 모델·α grid·`asr_topk`·도메인 용어집 경로를 지정한다.
데이터셋 경로(`dataset_path`)는 환경에 맞게 교체한다.

## 결과

honest 조건 POC (120 test, `whisper-large-v3`)에서 α=0.1 적용 시
**도메인 용어 recall +6.8%p, CER 동시 개선(−0.018), 환각 0** — 상세는
[`docs/poc_results_honest_v1.md`](docs/poc_results_honest_v1.md).

> 주의: 현재 코드는 domain-term metric을 longest-match/공백 무시 정책으로 고쳤다.
> 기존 POC 결과 문서의 term recall/precision 수치는 이전 단순 substring metric 기준이므로,
> CT2 이식 전 최신 metric으로 재측정해야 한다.
> 또한 기본 fusion mode는 과거 POC의 `topk` 근사에서 `topk_strict`로 바뀌었다.

> 주의: 학습/평가 데이터와 그 파생물(코퍼스·LM·전사)은 저장소에 포함하지 않는다 (`.gitignore: data/`).
