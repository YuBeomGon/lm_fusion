# GPT Review Follow-up — 2026-06-07

## 반영한 수정

- `fusion.mode=topk_strict` 추가: ASR top-k 밖 token을 `-inf`로 막고 top-k 내부에서만 LM 재랭킹한다.
- 기존 `topk` 모드는 과거 POC 재현용으로 유지했다.
- domain-term metric을 단순 `str.count()`에서 longest-match, non-overlapping, 공백 무시 정책으로 변경했다.
- n-best oracle term recall도 동일한 term matching 정책을 사용하도록 맞췄다.
- `max_audio_seconds`를 평가 전에 적용해 초과 오디오를 제외한다.
- 제외된 긴 오디오 수를 결과 JSONL의 `skipped_long_audio`에 기록한다.
- KenLM binary 파일명을 `kenlm.order` 설정에서 만든다.
- `device`, `dtype`, `num_beams`, `n_best`를 config로 분리했다.
- `--limit`가 남은 샘플 수보다 커도 실패하지 않게 했다.
- 최소 의존성 파일 `requirements.txt`를 추가했다.

## 결과 해석 변경

기존 POC 결과 문서의 term recall/precision은 이전 단순 substring metric 기준이다.
현재 코드는 metric과 기본 fusion mode가 바뀌었으므로, CT2 이식 게이트에는 최신 코드로 재측정한 결과를 사용해야 한다.

## 남은 후속

- 최신 metric과 `topk_strict` 기준으로 honest/ceiling alpha grid를 재실행한다.
- `full_vocab` 소규모 진단으로 `topk_strict`와의 차이를 확인한다.
- critical long-tail terms와 generic terms를 별도 리포트하는 지표를 추가한다.
- `kenlm.bin_dir`, `kenlm.lmplz_extra`를 학습 스크립트와 연결하거나 config 기반 wrapper를 만든다.
