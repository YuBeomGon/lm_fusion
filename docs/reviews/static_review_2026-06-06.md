# Static Review — 2026-06-06

## 범위

- 대상: `bpe_lm_fusion/`, `scripts/`, `configs/`, `README.md`, `docs/`
- 방식: 코드/문서 정적 리뷰
- 제외: 테스트 실행, 평가 실행, 모델 로딩, KenLM 바이너리 검증

## 요약

현재 프로젝트는 Whisper 디코딩에 BPE token-id KenLM을 `LogitsProcessor`로 결합하는 HuggingFace POC 구조다. 전체 흐름은 `build_corpus.py` -> `train_kenlm.sh` -> `run_eval.py`로 이어지며, 실험 아이디어와 코드 구조는 비교적 명확하다.

다만 현재 상태에서 POC 결과를 그대로 신뢰하거나 다음 단계인 CT2 이식 판단 근거로 쓰기에는 몇 가지 문제가 있다. 가장 중요한 리스크는 도메인 용어 지표 정의, top-k fusion의 점수 일관성, 30초 초과 오디오 처리, 문서/설정 불일치다.

## Critical

### 1. 도메인 용어 지표가 nested term을 중복 집계한다

- 위치: `bpe_lm_fusion/metrics.py:15`
- 위치: `configs/domain_terms.txt:8`
- 위치: `configs/domain_terms.txt:10`
- 위치: `configs/domain_terms.txt:32`
- 위치: `configs/domain_terms.txt:43`
- 위치: `configs/domain_terms.txt:55`

`term_recall_precision()`은 각 term에 대해 `ref.count(term)` / `hyp.count(term)`를 독립적으로 더한다. 그런데 용어집에는 `보험금지급사유`와 `보험금`, `보험계약체결`과 `보험계약`, `급여상해수술비`와 `상해`/`수술비`처럼 포함 관계가 많은 용어가 섞여 있다.

이 경우 한 번 등장한 긴 용어가 여러 짧은 용어의 hit로도 계산되어 `ref_total`, `hyp_total`, `matched`가 모두 부풀 수 있다. POC 핵심 지표인 domain-term recall/precision이 직접 영향을 받는다.

권장 수정:
- 용어를 길이 내림차순으로 정렬한 뒤 longest-match span 기준으로 집계한다.
- 겹치는 span은 한 번만 카운트한다.
- critical long-tail 지표와 generic term 지표를 분리한다.

### 2. 문서의 "공백 무시 substring 매칭"과 실제 구현이 다르다

- 위치: `configs/domain_terms.txt:2`
- 위치: `bpe_lm_fusion/metrics.py:20`
- 위치: `tests/test_metrics.py:15`

용어집 주석은 "공백 무시 substring 매칭"이라고 설명하지만 실제 코드는 단순 `str.count()`라 공백 변이를 무시하지 않는다. 테스트도 `후유 장해`를 `후유장해` 미일치로 기대한다.

한국어 STT에서는 띄어쓰기 변이가 흔하므로 이 불일치는 결과 해석에 큰 영향을 준다.

권장 수정:
- "공백 무시"가 정책이면 ref/hyp/term에서 공백 제거 후 span 매칭한다.
- "공백 민감"이 정책이면 용어집 주석과 문서를 수정한다.
- CER/WER용 정규화와 용어 매칭용 정규화는 목적이 다르므로 별도 함수로 분리한다.

### 3. top-k fusion이 non-top-k 토큰을 상대적으로 유리하게 만들 수 있다

- 위치: `bpe_lm_fusion/fusion_processor.py:39`
- 위치: `bpe_lm_fusion/fusion_processor.py:43`
- 위치: `bpe_lm_fusion/fusion_processor.py:52`

KenLM logprob는 보통 0 이하라 `alpha * lm_logprob`는 대부분 음수다. 현재 `mode=topk`는 ASR top-k 후보에만 이 음수 항을 더하고, top-k 밖 토큰은 원래 ASR 점수를 그대로 둔다.

즉 full-vocab shallow fusion의 근사라기보다는 "top-k 후보만 penalize하고 나머지는 neutral로 방치"하는 형태가 된다. alpha가 커지거나 LM 점수가 낮은 top-k 후보가 많으면 rank k+1 이후 토큰이 의도치 않게 올라올 수 있다.

권장 수정:
- 정확성 기준 평가는 `full_vocab`으로 최소 샘플에서 확인한다.
- `topk`를 유지하려면 non-top-k를 후보에서 제외할지, LM floor를 적용할지, 또는 top-k 내부 재랭킹으로만 쓸지 정책을 명확히 한다.
- 이 정책을 문서와 테스트에 고정한다.

### 4. 30초 초과 오디오 처리 정책이 구현되어 있지 않다

- 위치: `configs/poc.yaml:26`
- 위치: `docs/dataset_description.md:113`
- 위치: `docs/dataset_description.md:115`
- 위치: `bpe_lm_fusion/decode.py:31`
- 위치: `scripts/run_eval.py:44`

문서는 test에 30초 초과 chunk가 있고 "분할하거나 제외"해야 한다고 설명한다. 설정에도 `max_audio_seconds: 30`이 있다. 하지만 `run_eval.py`와 `FusionDecoder._features()`는 이 값을 사용하지 않는다.

Whisper feature extractor가 긴 오디오를 truncate하면, hyp는 앞 30초 기준인데 ref는 전체 transcript 기준이 되어 CER/WER가 왜곡될 수 있다. truncate하지 않더라도 Whisper 30초 window 계약과 맞지 않는다.

권장 수정:
- 평가 전에 `sampling_rate * max_audio_seconds` 기준으로 제외/분할 정책을 적용한다.
- 제외한 샘플 수를 결과 jsonl에 기록한다.
- split 평가라면 ref도 segment 단위로 맞춘다.

### 5. 용어집 출처 문서가 leakage 원칙과 충돌한다

- 위치: `docs/poc_results_honest_v1.md:9`
- 위치: `docs/dataset_description.md:92`
- 위치: `docs/dataset_description.md:94`

결과 문서는 용어집을 `hf_dataset(train+test)`에서 큐레이션했다고 적고, 데이터 문서는 test transcript에서 용어를 추출하면 안 된다고 적는다.

용어집은 LM 학습 데이터가 아니므로 모델 누수와는 다르지만, 평가 target을 test에서 보고 고르면 보고 지표가 선택 편향을 갖는다. 특히 POC의 핵심 성과가 domain-term recall이므로 이 충돌은 명확히 정리해야 한다.

권장 수정:
- `configs/domain_terms.txt`의 실제 생성 경위를 문서화한다.
- train-only/외부 도메인 사전 기반 용어집과 test-informed 용어집을 분리한다.
- POC 결과에는 어떤 용어집으로 계산했는지 명시한다.

## Major

### 6. KenLM order 설정이 평가 파일명에 반영되지 않는다

- 위치: `configs/poc.yaml:8`
- 위치: `configs/poc.yaml:9`
- 위치: `scripts/run_eval.py:33`
- 위치: `README.md:30`

설정에는 `kenlm.order`가 있지만 `run_eval.py`는 항상 `lm_<cond>_5g.binary`를 로드한다. order를 3으로 바꾸거나 다른 prefix로 학습하면 평가는 잘못된 파일을 찾는다.

권장 수정:
- `cfg["kenlm"]["order"]`로 파일명을 만든다.
- 가능하면 config에 명시적인 `lm_binary_template` 또는 `lm_binary_path`를 둔다.

### 7. `kenlm.bin_dir` / `kenlm.lmplz_extra` 설정이 학습 스크립트와 연결되어 있지 않다

- 위치: `configs/poc.yaml:10`
- 위치: `configs/poc.yaml:11`
- 위치: `scripts/train_kenlm.sh:2`
- 위치: `scripts/train_kenlm.sh:8`

KenLM 학습은 shell script positional argument로만 동작한다. config에 있는 `bin_dir`, `lmplz_extra`는 실제로 읽히지 않는다.

권장 수정:
- config 기반 학습 wrapper를 만들거나, README에서 script 인자를 config와 동기화해야 한다고 명시한다.

### 8. `run_eval --limit`가 데이터 길이보다 크면 실패할 수 있다

- 위치: `scripts/run_eval.py:26`
- 위치: `scripts/run_eval.py:27`
- 비교 위치: `scripts/run_rank_cdf.py:28`

`run_eval.py`는 `test.select(range(args.limit))`를 그대로 호출한다. `args.limit > len(test)`이면 out-of-range가 될 수 있다. `run_rank_cdf.py`는 `min(args.limit, len(...))`를 사용하므로 두 스크립트의 동작도 다르다.

권장 수정:
- `limit = min(args.limit, len(test))`를 적용한다.

### 9. baseline alpha=0 평가도 KenLM binary를 요구한다

- 위치: `scripts/run_eval.py:32`
- 위치: `scripts/run_eval.py:33`
- 위치: `scripts/run_eval.py:34`
- 위치: `bpe_lm_fusion/decode.py:43`

`alpha=0.0`은 fusion off지만, `run_eval.py`는 alpha loop 전에 항상 `KenlmScorer(lm_path)`를 만든다. 따라서 순수 baseline만 확인하고 싶어도 LM binary가 없으면 실행할 수 없다.

권장 수정:
- alpha가 0인 경우 scorer를 `None`으로 둔다.
- 또는 `--baseline-only` 경로를 분리한다.

### 10. 런타임 파라미터가 코드에 고정되어 있다

- 위치: `bpe_lm_fusion/decode.py:12`
- 위치: `bpe_lm_fusion/decode.py:13`
- 위치: `scripts/run_eval.py:48`

`device="cuda"`, `dtype=torch.float16`, `num_beams=5`, `n_best=5`가 코드에 고정되어 있다. 프로젝트 원칙상 모델/토글/임계값은 설정과 분리해야 한다.

권장 수정:
- config에 `runtime.device`, `runtime.dtype`, `decode.num_beams`, `decode.n_best`를 추가한다.
- CPU fallback이 필요 없다면 문서에 GPU 전제와 실패 조건을 명시한다.

### 11. `fusion.mode` 값 검증이 없다

- 위치: `configs/poc.yaml:16`
- 위치: `bpe_lm_fusion/fusion_processor.py:39`
- 위치: `bpe_lm_fusion/fusion_processor.py:42`

`mode`가 `full_vocab`이 아니면 모두 top-k 경로로 떨어진다. 오타가 나도 조용히 top-k로 실행된다.

권장 수정:
- `__init__`에서 `mode in {"topk", "full_vocab"}` 검증을 추가한다.
- `asr_topk <= vocab` 같은 기본 검증도 같이 둔다.

### 12. 패키징/의존성 파일이 없다

- 위치: `README.md:31`

루트에 `pyproject.toml`, `setup.py`, `requirements*.txt`가 없다. README도 `PYTHONPATH=.`를 붙여 실행한다. 재현 가능한 환경 생성과 기본 test/import 실행성이 약하다.

권장 수정:
- 최소 `pyproject.toml`에 package metadata와 dependencies를 둔다.
- `pytest`가 루트에서 package를 import할 수 있게 구성한다.

### 13. 문서의 현재 모델과 결과 모델이 섞여 있다

- 위치: `configs/poc.yaml:1`
- 위치: `README.md:39`
- 위치: `docs/poc_results_honest_v1.md:5`
- 위치: `docs/poc_results_honest_v1.md:50`

현재 config는 `openai/whisper-large-v3-turbo`지만 README의 결과 요약과 POC 결과 문서는 `openai/whisper-large-v3` 기준이다. POC 결과 문서에는 turbo 전환이 다음 단계라고 적혀 있다.

권장 수정:
- README에서 "현재 기본 실행 모델"과 "과거 POC 결과 모델"을 분리해 쓴다.
- turbo 재실험 전까지 v3 결과를 turbo 기대치처럼 읽히지 않게 표시한다.

### 14. POC 결과 문서에 미완성 값이 남아 있다

- 위치: `docs/poc_results_honest_v1.md:22`

WER 값이 `0.3?`로 남아 있다. 결과 문서가 게이트 판단 근거라면 불확실한 숫자는 제거하거나 원본 jsonl에서 정확히 복원해야 한다.

### 15. 구현 계획 문서가 현재 코드와 일부 불일치한다

- 위치: `docs/implementation_plan.md:3`
- 위치: `docs/implementation_plan.md:80`
- 위치: `docs/implementation_plan.md:859`
- 위치: `docs/implementation_plan.md:894`
- 위치: `bpe_lm_fusion/oracle.py:19`
- 위치: `tests/test_oracle.py:7`

`implementation_plan.md`는 아직 task checkbox가 비어 있는 구현 계획 형태다. 또 Task 10 예시는 raw oracle 함수가 `oracle_recall`을 반환한다고 쓰지만, 현재 코드는 `recall`을 반환하고 `run_eval.py`에서 prefix를 붙여 `oracle_recall`이 된다.

권장 수정:
- 구현 완료 후 계획 문서는 "완료된 구현 기록" 또는 "과거 계획"으로 명확히 표시한다.
- raw 함수 반환 키와 run_eval 출력 키를 문서에서 구분한다.

### 16. AGENTS 지침의 정본 문서가 없다

- 세션에 제공된 AGENTS 지침 기준: `docs/SSOT.md`

지침은 현재 정본 문서를 `docs/SSOT.md`로 둔다고 설명하지만, 현재 저장소에는 해당 파일이 없다. README와 여러 docs가 사실상 정본 역할을 나눠 갖고 있어 정책 충돌이 생기기 쉽다.

권장 수정:
- `docs/SSOT.md`를 만들고 현재 설계/실행 계약의 정본 위치를 고정한다.
- 나머지 문서는 요약과 링크 중심으로 정리한다.

## Minor

### 17. `KenlmScorer.most_common_first_token()`은 실제 common token이 아니다

- 위치: `bpe_lm_fusion/kenlm_scorer.py:63`
- 위치: `bpe_lm_fusion/kenlm_scorer.py:64`

테스트 편의용 함수가 항상 `0`을 반환한다. 하지만 corpus는 special token을 제외하고 `t<ID>`로 학습하므로 token 0이 common token이라는 보장이 없다.

권장 수정:
- 테스트 helper를 제거하거나, fixture corpus에서 실제 첫 토큰을 명시적으로 사용한다.

### 18. `domain_terms.load_terms()`가 파일 핸들을 직접 연다

- 위치: `bpe_lm_fusion/domain_terms.py:3`

작은 파일이라 실무상 영향은 작지만 context manager가 없다. 또한 normalization, dedupe, 빈 중복 제거 정책이 없다.

권장 수정:
- `with open(...)`을 사용한다.
- term normalization/dedupe를 loader에서 처리할지 metric에서 처리할지 정한다.

### 19. 모듈 헤더 형식이 프로젝트 원칙과 맞지 않는 파일이 있다

- 위치: `bpe_lm_fusion/domain_terms.py:1`
- 위치: `scripts/run_eval.py:1`
- 위치: `scripts/build_corpus.py:1`

AGENTS 지침은 Python 파일 최상단에 triple-quoted module header를 요구한다. 일부 파일은 `# path` 주석이 앞에 있거나, `domain_terms.py`처럼 헤더가 없다.

권장 수정:
- 포맷 정리 시 모듈 헤더 형식을 일괄 정리한다.

## 권장 우선순위

1. domain-term metric을 longest-match + 명시적 whitespace 정책으로 고정한다.
2. POC 결과 문서의 용어집 출처와 WER 미완성 값을 정리한다.
3. `max_audio_seconds` 처리 정책을 평가 코드에 반영한다.
4. top-k fusion의 non-top-k 처리 정책을 정하고 테스트로 고정한다.
5. KenLM order/path와 decode runtime 값을 config 기반으로 바꾼다.
6. `docs/SSOT.md`를 만들고 README/implementation plan/POC 결과 문서의 역할을 분리한다.

## 검증 상태

이번 리뷰에서는 사용자 요청에 따라 테스트를 실행하지 않았다. 위 내용은 파일 내용 기반의 정적 리뷰 결과다.
